"""Four-layer classroom annotation demonstrator. Run: streamlit run app.py"""
import re
import json
import pandas as pd
import streamlit as st
import spacy

st.set_page_config(page_title='LinguAnnotate 4D', page_icon='📚', layout='wide')
import streamlit.components.v1 as components
from spacy import displacy

PENN = {'CC':'Coordinating conjunction','CD':'Cardinal number','DT':'Determiner','EX':'Existential there','FW':'Foreign word','IN':'Preposition / subordinating conjunction','JJ':'Adjective','JJR':'Comparative adjective','JJS':'Superlative adjective','LS':'List item marker','MD':'Modal','NN':'Singular/common noun','NNS':'Plural noun','NNP':'Singular proper noun','NNPS':'Plural proper noun','PDT':'Predeterminer','POS':'Possessive ending','PRP':'Personal pronoun','PRP$':'Possessive pronoun','RB':'Adverb','RBR':'Comparative adverb','RBS':'Superlative adverb','RP':'Particle','SYM':'Symbol','TO':'to','UH':'Interjection','VB':'Base-form verb','VBD':'Past-tense verb','VBG':'Gerund / present participle','VBN':'Past participle','VBP':'Non-third-person present verb','VBZ':'Third-person singular present verb','WDT':'Wh-determiner','WP':'Wh-pronoun','WP$':'Possessive wh-pronoun','WRB':'Wh-adverb', '.':'Sentence-final punctuation',',':'Comma',':':'Colon / semicolon', '$':'Currency symbol',"``":'Opening quote',"''":'Closing quote','-LRB-':'Opening bracket','-RRB-':'Closing bracket','#':'Number sign'}
UD = {'acl':'Clausal modifier of noun','advcl':'Adverbial clause modifier','advmod':'Adverbial modifier','amod':'Adjectival modifier','appos':'Appositional modifier','aux':'Auxiliary','case':'Case marking','cc':'Coordinating conjunction','ccomp':'Clausal complement','clf':'Classifier','compound':'Compound','conj':'Conjunct','cop':'Copula','csubj':'Clausal subject','dep':'Unspecified dependency','det':'Determiner','discourse':'Discourse element','dislocated':'Dislocated element','expl':'Expletive','fixed':'Fixed multiword expression','flat':'Flat multiword expression','goeswith':'Goes with','iobj':'Indirect object','list':'List','mark':'Marker','nmod':'Nominal modifier','nsubj':'Nominal subject','nummod':'Numeric modifier','obj':'Object','obl':'Oblique nominal','orphan':'Orphan','parataxis':'Parataxis','punct':'Punctuation','reparandum':'Overridden disfluency','root':'Root','vocative':'Vocative','xcomp':'Open clausal complement'}
# Classroom approximations derived from the supplied verb-group mapping; not VerbNet class/frame inference.
VERB_GROUPS = {
 'Giving':('give lend hand pass', 'Agent','Theme','Recipient'),
 'Sending':('send mail dispatch','Agent','Theme','Destination'),
 'Breaking':('break shatter crack','Agent/Causer','Patient',None),
 'Cutting':('cut slice chop','Agent','Patient','Instrument'),
 'Creating':('make build construct','Agent','Product','Material'),
 'Movement':('walk run travel','Agent/Theme',None,'Path/Destination'),
 'Placement':('put place position','Agent','Theme','Destination'),
 'Removing':('remove extract','Agent','Theme','Source'),
 'Emotions':('fear love hate','Experiencer','Stimulus',None),
 'Emotional causation':('frighten annoy amuse','Stimulus/Causer','Experiencer',None),
 'Perception':('see hear notice','Experiencer','Stimulus',None),
 'Communication':('say tell explain','Agent','Topic','Recipient'),
 'Buying':('buy purchase','Agent','Theme','Asset/Source'),
 'Selling':('sell trade','Agent','Theme','Recipient/Asset'),
 'Cooking':('cook bake prepare','Agent','Patient/Product','Instrument'),
 'Change of state':('melt freeze dry','Patient','Result',None),
 'Learning':('learn study understand','Experiencer/Agent','Topic/Theme',None),
 'Helping':('help assist support','Agent','Beneficiary/Theme',None),
 'Searching':('search seek investigate','Agent','Theme/Location',None),
 'Measuring':('measure weigh calculate','Agent','Theme','Value')
}
LOOKUP = {v:(group,subject,obj,extra) for group,(verbs,subject,obj,extra) in VERB_GROUPS.items() for v in verbs.split()}
ROLE_GLOSS = {'Agent':'intentional doer','Theme':'entity involved or moved','Patient':'affected entity','Recipient':'receiver','Experiencer':'one who feels or perceives','Stimulus':'trigger of a feeling','Instrument':'tool used','Destination':'endpoint','Source':'origin','Product':'thing created','Beneficiary':'one who benefits','Topic':'subject matter','Asset':'resource exchanged','Causer':'non-intentional cause','Value':'measurement'}
# Connective-to-sense mappings are intentionally conservative; ambiguous cues are marked provisional.
CONNECTIVES = [
 ('because','Contingency.Cause.Reason','subordinate'),('although','Comparison.Concession.Arg1-as-denier','subordinate'),('though','Comparison.Concession.Arg1-as-denier','subordinate'),('even though','Comparison.Concession.Arg1-as-denier','subordinate'),
 ('if','Contingency.Condition.Arg2-as-cond','subordinate'),('unless','Contingency.Negative-condition.Arg2-as-negCond','subordinate'),
 ('before','Temporal.Asynchronous.Precedence','subordinate'),('after','Temporal.Asynchronous.Succession','subordinate'),('while','Temporal.Synchronous','subordinate'),('when','Temporal.Synchronous','subordinate'),
 ('whereas','Comparison.Contrast','subordinate'),('since','Contingency.Cause.Reason','subordinate'),
 ('however','Comparison.Contrast','between'),('therefore','Contingency.Cause.Result','between'),('moreover','Expansion.Conjunction','between'),('furthermore','Expansion.Conjunction','between'),('similarly','Comparison.Similarity','between'),('for example','Expansion.Instantiation.Arg2-as-instance','between'),('in other words','Expansion.Equivalence','between'),
 ('but','Comparison.Contrast','between'),('and','Expansion.Conjunction','between'),('or','Expansion.Disjunction','between')]
CONNECTIVES.sort(key=lambda x:len(x[0]), reverse=True)
SENSES = ['Temporal.Synchronous','Temporal.Asynchronous.Precedence','Temporal.Asynchronous.Succession','Contingency.Cause.Reason','Contingency.Cause.Result','Contingency.Cause.NegResult','Contingency.Cause+Belief.Reason+Belief','Contingency.Cause+Belief.Result+Belief','Contingency.Cause+SpeechAct.Reason+SpeechAct','Contingency.Cause+SpeechAct.Result+SpeechAct','Contingency.Condition.Arg1-as-cond','Contingency.Condition.Arg2-as-cond','Contingency.Condition+SpeechAct','Contingency.Negative-condition.Arg1-as-negCond','Contingency.Negative-condition.Arg2-as-negCond','Contingency.Negative-condition+SpeechAct','Contingency.Purpose.Arg1-as-goal','Contingency.Purpose.Arg2-as-goal','Comparison.Concession.Arg1-as-denier','Comparison.Concession.Arg2-as-denier','Comparison.Concession+SpeechAct','Comparison.Contrast','Comparison.Similarity','Expansion.Conjunction','Expansion.Disjunction','Expansion.Equivalence','Expansion.Exception.Arg1-as-excpt','Expansion.Exception.Arg2-as-excpt','Expansion.Instantiation.Arg1-as-instance','Expansion.Instantiation.Arg2-as-instance','Expansion.Level-of-detail.Arg1-as-detail','Expansion.Level-of-detail.Arg2-as-detail','Expansion.Manner.Arg1-as-manner','Expansion.Manner.Arg2-as-manner','Expansion.Substitution.Arg1-as-subst','Expansion.Substitution.Arg2-as-subst']

@st.cache_resource
def get_nlp():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        return None

def phrase(token):
    """Get a readable dependency subtree span; can over-include in complex sentences."""
    words = list(token.subtree)
    if not words: return token.text
    return token.doc[min(t.i for t in words):max(t.i for t in words)+1].text

# VerbNet XML is bundled with the repository; students do not upload resources.
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

ROLE_EXPLANATIONS = {
    'Agent':'An intentional actor or initiator of the event.',
    'Patient':'The entity affected by the event.',
    'Theme':'An entity involved in or undergoing the event.',
    'Instrument':'A tool or means used to carry out the event.',
    'Experiencer':'The participant experiencing a perception or psychological state.',
    'Stimulus':'The participant or situation triggering an experience.',
    'Recipient':'The receiver of an entity or message.',
    'Destination':'The endpoint of movement or transfer.',
    'Source':'The starting point of movement or transfer.',
    'Result':'The resulting entity or state.',
    'Location':'The location associated with the event.',
    'Beneficiary':'The participant who benefits from the event.',
    'Topic':'The content or subject of an event.',
    'Cause':'An event or entity causing an effect.',
    'Co-Agent':'An additional actor involved in the event.',
    'Asset':'A resource exchanged in a transaction.',
}

def phrase(token):
    ids=[x.i for x in token.subtree]
    return token.doc[min(ids):max(ids)+1].text if ids else token.text

@st.cache_resource(show_spinner='Loading bundled VerbNet XML lexicon…')
def load_verbnet():
    archive=Path(__file__).resolve().parent/'verbnet_data.zip'
    if not archive.is_file():
        return {}, f'Missing VerbNet archive at {archive}. Ensure the ZIP is committed in the same GitHub folder as this running Python script.'
    index={}
    try:
        z = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        return {}, f'Cannot open verbnet_data.zip: {exc}'
    with z:
        for name in z.namelist():
            if not name.endswith('.xml'):continue
            try: root=ET.fromstring(z.read(name))
            except ET.ParseError:continue
            if root.tag!='VNCLASS':continue
            def walk(node, inherited_frames=(), inherited_roles=()):
                cid=node.get('ID','')
                roles=tuple(dict.fromkeys([*inherited_roles,*[r.get('type','') for r in node.findall('./THEMROLES/THEMROLE')]]))
                frames=tuple(inherited_frames)+tuple(parse_frame(f) for f in node.findall('./FRAMES/FRAME'))
                for m in node.findall('./MEMBERS/MEMBER'):
                    lemma=m.get('name','').lower().strip()
                    if lemma:index.setdefault(lemma,[]).append({'classid':cid,'frames':frames,'roles':roles})
                for sub in node.findall('./SUBCLASSES/VNSUBCLASS'):walk(sub,frames,roles)
            walk(root)
    return index, f'Loaded verbnet_data.zip: {len(index)} verb lemmas available.'

def parse_frame(frame):
    desc=frame.find('DESCRIPTION')
    syntax=[]
    for el in frame.findall('./SYNTAX/*'):
        if el.tag in ('NP','VERB','PREP','LEX','PP','ADJ','ADV','S','VP'):
            syntax.append({'pos':el.tag,'value':el.get('value','')})
    examples=[x.text for x in frame.findall('./EXAMPLES/EXAMPLE') if x.text]
    return {'name':desc.get('primary','') if desc is not None else '',
            'secondary':desc.get('secondary','') if desc is not None else '',
            'syntax':syntax,'example':examples[0] if examples else ''}

def participants_for(verb):
    parts=[]
    for c in verb.children:
        dep=c.dep_.lower()
        if dep in ('nsubj','csubj','nsubjpass','csubjpass'):
            kind='subject'
        elif dep in ('dobj','obj','attr','oprd','acomp'):
            kind='object'
        elif dep in ('dative','iobj'):
            kind='iobj'
        elif dep in ('prep','agent'):
            kind='pp'
            if not any(x.dep_ in ('pobj','obj') for x in c.children):continue
        else:continue
        subtree=list(c.subtree)
        parts.append({'text':phrase(c),'kind':kind,'prep':c.text.lower() if kind=='pp' else '',
                      'dep':dep,'start':min(x.i for x in subtree),'end':max(x.i for x in subtree)+1})
    # Passive constructions need a separate role alignment; do not silently invert arguments.
    return sorted(parts,key=lambda p:p['start'])

def align_frame(frame,parts):
    syntax=frame['syntax']
    if not any(s['pos']=='VERB' for s in syntax):return None
    subjects=[p for p in parts if p['kind']=='subject']
    objects=[p for p in parts if p['kind']=='object']
    indirects=[p for p in parts if p['kind']=='iobj']
    pps=[p for p in parts if p['kind']=='pp']
    if not subjects:return None
    # Distinguish core NP slots from NP complements of PREP/LEX/PP.
    core=[];preps=[];last_prep=None
    for s in syntax:
        if s['pos'] in ('PREP','LEX','PP'):
            last_prep=s['value'].lower().split() if s['value'] else []
        elif s['pos']=='NP':
            if last_prep is None:core.append(s['value'])
            else:preps.append((last_prep,s['value']));last_prep=None
        elif s['pos'] not in ('VERB',):
            # Frame includes a complement we cannot confidently align.
            return None
    if len(core)<1 or not core[0]:return None
    if len(core)!=1+len(objects)+len(indirects):return None
    if len(preps)>len(pps):return None
    mapping=[(subjects[0],core[0],'VerbNet subject NP frame slot')]
    remaining=sorted(objects+indirects,key=lambda p:p['start'])
    for p,role in zip(remaining,core[1:]):
        if not role:return None
        mapping.append((p,role,'VerbNet core NP frame slot'))
    used=set()
    for allowed,role in preps:
        found=None
        for j,p in enumerate(pps):
            if j not in used and (not allowed or p['prep'] in allowed):found=j;break
        if found is None or not role:return None
        used.add(found)
        mapping.append((pps[found],role,'VerbNet preposition + NP frame slot'))
    # Prefer matching more of the observed sentence and specifically licensed prepositions.
    unexplained=len(pps)-len(used)
    score=20*len(mapping)+15*len(preps)-12*unexplained
    if unexplained==0:score+=20
    return score,mapping

def analyze_verbnet(doc):
    index,source=load_verbnet()
    rows=[];summaries=[]
    verbs=[t for t in doc if t.pos_=='VERB' and t.dep_ not in ('aux','auxpass')]
    for v in verbs:
        lemma=v.lemma_.lower()
        parts=participants_for(v)
        candidates=index.get(lemma,[])
        passive=any(p['dep'] in ('nsubjpass','csubjpass') for p in parts)
        matches=[]
        if not passive:
            for candidate in candidates:
                for frame in candidate['frames']:
                    aligned=align_frame(frame,parts)
                    if aligned is not None:
                        score,mapping=aligned
                        # Prefer semantically identifiable class names, but never infer meaning from class name alone.
                        # For e.g. cut-21.1-1, prefer its inherited cut-21.1 frames.
                        matches.append((score,candidate,frame,mapping))
        # Prefer a class whose name corresponds to the predicate lemma when frames tie.
        # This is a lexical tie-break, not trained word-sense disambiguation.
        matches.sort(key=lambda m:(-m[0], -int(m[1]['classid'].split('-')[0].lower()==lemma), m[1]['classid'],m[2]['name']))
        best=matches[0] if matches else None
        selected=best[1]['classid'] if best else 'None'
        frame=best[2] if best else None
        ambiguity=(len({m[1]['classid'] for m in matches if m[0]==best[0]})>1) if best else False
        status=('Needs verification: multiple equally scored classes' if ambiguity else
                'Frame-compatible suggestion — review semantics' if best else
                'No compatible frame — review manually')
        summaries.append({'Predicate':v.text,'Lemma':lemma,
            'Candidate classes':', '.join(c['classid'] for c in candidates[:12]) or 'None',
            'Selected candidate':selected,'Candidate frame':frame['name'] if frame else 'None',
            'Frame example':frame['example'] if frame else '',
            'Lexicon source':source,'Match status':status,
            'Interpretation':'Frame roles are extracted from the bundled XML; selecting the intended verb sense is not guaranteed.'})
        bindings={(p['start'],p['end']):(role,evidence) for p,role,evidence in (best[3] if best else [])}
        for p in parts:
            role,evidence=bindings.get((p['start'],p['end']),('Unresolved','No compatible XML frame slot'))
            rows.append({'Predicate':v.text,'Lemma':lemma,'VerbNet class (candidate)':selected,
                'Frame':frame['name'] if frame else '', 'Expression':p['text'],
                'Grammatical evidence':p['kind']+' / '+p['dep'],
                'Thematic role (suggested)':role,'Corrected role':role,
                'Explanation':evidence+'. '+ROLE_EXPLANATIONS.get(role,''),
                'Reliability':status if role!='Unresolved' else 'Unresolved — review manually',
                'Reviewer notes':''})
    return rows,summaries,source

def trim(s): return s.strip(' \n\t,;:.!?')

def discourse(text):
    found=[]
    for cue,sense,kind in CONNECTIVES:
        for match in re.finditer(r'(?<!\w)'+re.escape(cue)+r'(?!\w)',text,re.I):
            a,b=text[:match.start()],text[match.end():]
            if not a.strip() and not b.strip():continue
            if kind=='subordinate':
                if a.strip():
                    # For a postposed subordinate clause, its span is Arg2.
                    arg1=trim(a); arg2=trim(b)
                else:
                    parts=re.split(r'[,;]',b,maxsplit=1)
                    if len(parts)<2: continue
                    arg2=trim(parts[0]);arg1=trim(parts[1])
            else:
                arg1=trim(a);arg2=trim(b)
            if not arg1 or not arg2:continue
            # Directional PDTB labels need manual verification. These are cue-based defaults.
            found.append({'Arg1':arg1,'Connective':match.group(),'Arg2':arg2,'Relation type':'Explicit','Sense suggestion':sense,'Level 1':sense.split('.')[0],'Method':'Connective rule (not a trained PDTB parser)','Review':'Required: verify argument spans, sense and direction'})
        if found: break
    return found

def dot_syntax(doc):
    lines=['digraph G {','rankdir=TB;','node [shape=box, style="rounded,filled", fillcolor="#edf3fa", fontname="Arial"];']
    for t in doc:
        lines.append(f'n{t.i} [label={json.dumps(t.text+"\\n"+t.dep_)}];')
        if t.head != t: lines.append(f'n{t.head.i} -> n{t.i} [label={json.dumps(t.dep_)}];')
    return '\n'.join(lines+['}'])

def dot_pos(rows):
    lines=['digraph POS {','rankdir=LR;','node [shape=box, style="rounded,filled", fillcolor="#eaf2ff", fontname="Arial"];']
    for i,r in enumerate(rows):
        lines.append(f'w{i} [label={json.dumps(r["Token"]+"\n"+r["Penn tag"])}];')
        if i: lines.append(f'w{i-1} -> w{i} [color="#a3b0c2", arrowsize=0.5];')
    return '\n'.join(lines+['}'])

def dot_roles(rows):
    lines=['digraph Roles {','rankdir=LR;','node [shape=box,style="rounded,filled",fillcolor="#eaf5ef",fontname="Arial"];']
    verbs=list(dict.fromkeys(r['Predicate'] for r in rows))
    for i,v in enumerate(verbs):
        lines.append(f'v{i} [label={json.dumps("VERB: "+v)},fillcolor="#ffe9cc"];')
        for j,r in enumerate(x for x in rows if x['Predicate']==v):
            key=f'r{i}_{j}'
            lines.append(f'{key} [label={json.dumps(r["Expression"]+"\n"+r["Thematic role (suggested)"])}];')
            lines.append(f'v{i} -> {key} [label={json.dumps(r["Thematic role (suggested)"])}];')
    return '\n'.join(lines+['}'])

UD_NORMALIZE={'ROOT':'root','dobj':'obj','pobj':'obl','prep':'case','nsubjpass':'nsubj:pass','csubjpass':'csubj:pass','auxpass':'aux:pass','poss':'nmod:poss','npadvmod':'obl','acomp':'xcomp','attr':'xcomp','relcl':'acl:relcl','agent':'obl:agent','prt':'compound:prt','oprd':'xcomp','neg':'advmod','dative':'iobj','predet':'det:predet'}
def ud_equivalent(label):
    return UD_NORMALIZE.get(label,label if label in UD else 'review manually')

def dot_disc(row):
    return 'digraph D {rankdir=TB; node [shape=box,style="rounded,filled",fillcolor="#eef5ff"]; root [label='+json.dumps(row['Sense suggestion'])+']; root -> arg1; root -> conn; root -> arg2; arg1 [label='+json.dumps('Arg1: '+row['Arg1'])+']; conn [label='+json.dumps('Connective: '+row['Connective'])+']; arg2 [label='+json.dumps('Arg2: '+row['Arg2'])+'];}'

st.title('📚 LinguAnnotate 4D — Four-Layer Annotation Studio')
st.caption('Penn Treebank POS • UD-style syntax • bundled VerbNet XML frames • PDTB 3.0-inspired discourse relations')
st.warning('Accuracy note: Penn POS and UD-style dependencies are spaCy predictions; UD labels are approximate mappings. VerbNet roles use actual XML frame slots but verb-sense selection is not a trained model and can be ambiguous. PDTB uses connective rules. Verify research annotations.')
nlp=get_nlp()
if nlp is None:
    st.error('The English spaCy model is missing. Run: python -m spacy download en_core_web_sm  (then restart Streamlit).')
    st.stop()

samples=['Sara gave Ali a book because he needed it.','Although Sara was tired, she completed her assignment.','Ali cut the paper with scissors.','Sara stayed home because she was ill.','The storm broke the window.','Sara sent a parcel to Lahore.']
example=st.selectbox('Sample sentences', ['Write my own']+samples)
text=st.text_area('Paste an English sentence or short paragraph',value='' if example=='Write my own' else example,height=115,placeholder='Sara gave Ali a book because he needed it.')
with st.expander('VerbNet lexical source',expanded=False):
    st.write(load_verbnet()[1])
    st.caption('The bundled XML is loaded automatically. No student upload is required. This edition uses frame matching, not the pretrained VerbNetParser neural models.')
if st.button('Analyse all four layers',type='primary'):
    if not text.strip(): st.info('Please enter some text.');st.stop()
    if len(text)>5000: st.error('Please limit input to 5,000 characters.');st.stop()
    doc=nlp(text)
    posrows=[{'Token':t.text,'Lemma':t.lemma_,'Penn tag':t.tag_,'Full name':PENN.get(t.tag_,'Model-specific tag'),'Universal POS':t.pos_} for t in doc]
    synrows=[{'Token':t.text,'Head':t.head.text if t.head!=t else 'ROOT','spaCy dependency':t.dep_,'Approx. UD equivalent':ud_equivalent(t.dep_),'Meaning':UD.get(ud_equivalent(t.dep_).split(':')[0], 'Review with UD guidelines'),'Head index':t.head.i} for t in doc]
    st.session_state.pop('corrected_semantic',None)
    semrows,sem_summary,model_status=analyze_verbnet(doc);discrows=discourse(text)
    st.session_state['results']={'pos':posrows,'syntax':synrows,'semantic':semrows,'semantic_summary':sem_summary,'model_status':model_status,'discourse':discrows,'syntax_dot':dot_syntax(doc),'text':text}

if 'results' in st.session_state:
    r=st.session_state['results']
    tabs=st.tabs(['1. Penn POS','2. UD syntax','3. VerbNet thematic roles','4. PDTB discourse','5. Export / reference'])
    with tabs[0]:
        st.subheader('Penn Treebank part-of-speech tags')
        st.markdown('**Visual POS sequence** — each word is paired with its predicted Penn tag.')
        st.graphviz_chart(dot_pos(r['pos']),use_container_width=True)
        st.dataframe(pd.DataFrame(r['pos']),use_container_width=True,hide_index=True)
        st.markdown('**How to read the results:** Token is the word; lemma is its dictionary form; Penn tag is its grammatical category; full name explains the tag. These are automatic predictions, not manually verified gold labels.')
        st.caption('spaCy tag_ gives Penn-style tags. The supplied PDF includes rare and ambiguous slash tags that this model does not explicitly predict.')
    with tabs[1]:
        st.subheader('Universal Dependencies-style syntax')
        st.dataframe(pd.DataFrame(r['syntax']),use_container_width=True,hide_index=True)
        st.markdown('**How to read the table:** Each token depends on a head word. The arrow in the tree points from head to dependent; the label describes the grammatical relationship. spaCy uses its own labels; the UD column is an approximate mapping, not a UD-trained model.')
        st.subheader('Visual dependency tree')
        st.graphviz_chart(r['syntax_dot'],use_container_width=True)
        st.caption('spaCy dependency labels can differ from the canonical UD inventory (e.g., dobj versus obj). This view does not claim a fully UD-conformant parse.')
    with tabs[2]:
        st.subheader('VerbNet — candidate classes, frames and thematic roles')
        st.caption('Bundled VerbNet XML classes and frames. This is a frame-constrained automatic suggestion, not a trained semantic-role model.')
        st.info('Lexicon: '+r.get('model_status','Not loaded'))
        if r.get('semantic_summary'):
            st.markdown('**Predicate and class lookup**')
            st.dataframe(pd.DataFrame(r['semantic_summary']),use_container_width=True,hide_index=True)
            for entry in r['semantic_summary']:
                with st.expander('Why was '+entry['Predicate']+' analysed this way?'):
                    st.write('**Lemma:**',entry['Lemma'])
                    st.write('**Candidate class:**',entry['Selected candidate'])
                    st.write('**Frame:**',entry['Candidate frame'])
                    st.write('**Assessment:**',entry['Match status'])
                    st.write(entry['Interpretation'])
        if r['semantic']:
            st.markdown('**Visual thematic-role map**')
            st.markdown('**Editable annotations** — correct any suggested role and add a justification. Changes are retained for export.')
            edit_df=pd.DataFrame(st.session_state.get('corrected_semantic',r['semantic']))
            role_choices=sorted(set(ROLE_EXPLANATIONS)|set(ROLE_GLOSS)|{'Unresolved','Other'})
            edited=st.data_editor(edit_df,use_container_width=True,hide_index=True,num_rows='fixed',key='role_editor',
                disabled=[c for c in edit_df.columns if c not in ('Corrected role','Reviewer notes')],
                column_config={'Corrected role':st.column_config.SelectboxColumn('Corrected role',options=role_choices,required=True),
                               'Reviewer notes':st.column_config.TextColumn('Reviewer notes')})
            st.session_state['corrected_semantic']=edited.to_dict('records')
            st.markdown('**Visual thematic-role map (corrected roles)**')
            vis=edited.to_dict('records')
            for x in vis:x['Thematic role (suggested)']=x['Corrected role']
            st.graphviz_chart(dot_roles(vis),use_container_width=True)
            st.markdown('**Participant-by-participant explanation**')
            st.dataframe(edited,use_container_width=True,hide_index=True)
        else:
            st.info('No grammatical participants could be assigned. Inspect the class inventory and annotate manually.')
        st.warning('A matching class and NP order do not establish the correct semantic sense. Passive clauses, complex complements, prepositional roles, and coreference require manual review. Do not use these suggestions as validated research annotations.')
        with st.expander('Role definitions'):
            st.dataframe(pd.DataFrame([{'Role':k,'Simple meaning':v} for k,v in {**ROLE_GLOSS,**ROLE_EXPLANATIONS}.items()]),hide_index=True,use_container_width=True)
    with tabs[3]:
        st.subheader('PDTB 3.0-inspired discourse analysis')
        if r['discourse']:
            for i,row in enumerate(r['discourse']):
                st.markdown(f'**Relation {i+1} — suggested, requires checking**')
                st.table(pd.DataFrame([{'Element':k,'Result':v} for k,v in row.items()]))
                st.graphviz_chart(dot_disc(row),use_container_width=True)
                st.markdown('**Interpretation:** Arg1 and Arg2 are the two connected spans. The connective is the signalling expression. Explicit means a connective is visible. The sense is a *suggested* meaning; directional subtypes require manual checking. This is not a trained PDTB 3.0 parser.')
                with st.expander('Choose a corrected PDTB 3.0 sense'):
                    corrected=st.selectbox('Sense inventory',SENSES,index=SENSES.index(row['Sense suggestion']) if row['Sense suggestion'] in SENSES else 0,key=f'sense_{i}')
                    st.caption('This is a manual correction, not an automatic model prediction.')
                    st.write('Selected sense:',corrected)
        else:
            st.info('No supported explicit connective detected. Implicit, AltLex, AltLexC, EntRel, Hypophora and NoRel require manual annotation in this prototype.')
        st.caption('Some connectives are ambiguous (e.g. since, while, but). Arg1/Arg2 boundaries and directional labels are approximations.')
    with tabs[4]:
        st.subheader('Download the annotations')
        combined={'input':r['text'],'penn_pos':r['pos'],'ud_style_syntax':r['syntax'],'verbnet_candidate_frames':r.get('semantic_summary',[]),'verbnet_suggested_roles':r['semantic'],'verbnet_corrected_roles':st.session_state.get('corrected_semantic',r['semantic']),'pdtb_rule_suggestions':r['discourse'],'method_warning':'VerbNet frame alignment and sense selection are tentative; no trained VerbNet SRL model is included. PDTB is rule-based.'}
        st.download_button('Download all results (JSON)',json.dumps(combined,indent=2,ensure_ascii=False),file_name='corpus_annotation.json',mime='application/json')
        for key,name in [('pos','penn_pos'),('syntax','ud_syntax'),('semantic','thematic_roles'),('discourse','pdtb_discourse')]:
            st.download_button(f'Download {name}.csv',pd.DataFrame(st.session_state.get('corrected_semantic',r[key]) if key=='semantic' else r[key]).to_csv(index=False),file_name=f'{name}.csv',mime='text/csv',key='dl_'+key)
        st.download_button('Download dependency tree (.dot)',r['syntax_dot'],file_name='dependency_tree.dot')
        st.markdown('**Official references:** [Penn Treebank](https://www.cis.upenn.edu/~treebank/) · [UD](https://universaldependencies.org/u/dep/) · [VerbNet](https://verbs.colorado.edu/verb-index/) · [PDTB 3.0](https://catalog.ldc.upenn.edu/LDC2019T05)')
else:
    st.info('Enter a sentence and click **Analyse all four layers** to begin.')
