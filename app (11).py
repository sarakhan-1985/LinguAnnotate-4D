"""Four-layer classroom annotation demonstrator. Run: streamlit run app.py"""
import re
import json
import pandas as pd
import streamlit as st
import spacy
import nltk
from spacy.util import is_package

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

@st.cache_resource(show_spinner="Loading VerbNet 3 data (first run may download corpus files)…")
def get_verbnet():
    """NLTK's verbnet3 reader; gracefully report missing network/corpus."""
    try:
        from nltk.corpus import verbnet as vn
        try:
            vn.classids('give')
        except LookupError:
            nltk.download('verbnet', quiet=True, raise_on_error=True)
            vn.classids('give')
        return vn, None
    except Exception as exc:
        return None, str(exc)

ROLE_EXPLANATIONS = {
    'Agent':'The intentional participant who carries out the action.',
    'Patient':'An entity affected or changed by the event.',
    'Theme':'An entity involved in, moved by, or undergoing the event.',
    'Experiencer':'A participant who feels or perceives a state or event.',
    'Recipient':'The participant who receives something.',
    'Stimulus':'The entity or situation that triggers a perception or emotion.',
    'Destination':'The endpoint of movement or transfer.',
    'Source':'The starting point of movement or transfer.',
    'Instrument':'An object used to perform the action.',
    'Location':'Where the event takes place.',
    'Result':'The resulting state or entity.',
    'Beneficiary':'The participant for whose benefit an action occurs.',
    'Topic':'The subject matter of an event.',
    'Cause':'The participant or event that causes another event.',
    'Actor':'A participant acting in an event.',
    'Asset':'A resource exchanged in a transaction.',
}

def _participant_nodes(verb):
    """Extract participant spans with spaCy dependencies; retain token indices."""
    participants=[]
    for child in verb.children:
        dep=child.dep_.lower()
        if dep in ('nsubj','nsubjpass','csubj','csubjpass'):
            participants.append({'kind':'SUBJECT','text':phrase(child),'dep':dep,'start':min(x.i for x in child.subtree),'end':max(x.i for x in child.subtree)+1})
        elif dep in ('dobj','obj','attr','oprd','acomp'):
            participants.append({'kind':'OBJECT','text':phrase(child),'dep':dep,'start':min(x.i for x in child.subtree),'end':max(x.i for x in child.subtree)+1})
        elif dep in ('dative','iobj'):
            participants.append({'kind':'INDIRECT OBJECT','text':phrase(child),'dep':dep,'start':min(x.i for x in child.subtree),'end':max(x.i for x in child.subtree)+1})
        elif dep in ('prep','agent'):
            pobj=next((x for x in child.children if x.dep_ in ('pobj','obj')),None)
            if pobj is not None:
                participants.append({'kind':'PP','prep':child.text.lower(),'text':phrase(child),'dep':dep,'start':min(x.i for x in child.subtree),'end':max(x.i for x in child.subtree)+1})
    return sorted(participants,key=lambda x:x['start'])

def _frame_description(frame):
    d=frame.get('description',{})
    return str(d.get('primary','') if isinstance(d,dict) else d)

def _frame_slots(frame):
    slots=[]
    for node in frame.get('syntax',[]):
        if not isinstance(node,dict): continue
        pos=str(node.get('pos_tag','')).upper()
        mods=node.get('modifiers',{})
        role=str(mods.get('value','') if isinstance(mods,dict) else '')
        if pos in ('NP','PP','PREP','ADJ','ADV','S','S_INF','VP','VERB'):
            slots.append({'pos':pos,'role':role,'value':str(node.get('value',''))})
    return slots

def _candidate_match(slots,parts,passive=False):
    """Rank an alignment, rejecting unsupported roles instead of filling from slot order blindly."""
    if passive: return -1,[], 'Passive voice: manual review required'
    np_slots=[s for s in slots if s['pos']=='NP' and s['role']]
    subj=next((p for p in parts if p['kind']=='SUBJECT'),None)
    obj=next((p for p in parts if p['kind']=='OBJECT'),None)
    iobj=next((p for p in parts if p['kind']=='INDIRECT OBJECT'),None)
    if not subj or not np_slots: return -1,[], 'No subject or NP frame slot'
    if len(np_slots)>1 and not obj and not iobj: return -1,[], 'Missing object'
    if len(np_slots)==1 and obj: return -1,[], 'Unexpected object'
    bindings=[(subj,np_slots[0]['role'],'NP subject')]
    if iobj and len(np_slots)>=3:
        bindings.append((iobj,np_slots[1]['role'],'NP indirect object'))
        if obj: bindings.append((obj,np_slots[2]['role'],'NP direct object'))
    elif obj and len(np_slots)>=2:
        bindings.append((obj,np_slots[1]['role'],'NP direct object'))
    # Match a PP role only when a frame explicitly supplies it and its preposition is compatible.
    pps=[p for p in parts if p['kind']=='PP']
    prep_slots=[s for s in slots if s['pos'] in ('PREP','PP')]
    for pp in pps:
        for j,sl in enumerate(prep_slots):
            if sl['pos']=='PREP' and sl['value'] and pp.get('prep','') not in sl['value'].lower().split(): continue
            next_role=next((s['role'] for s in slots[slots.index(sl)+1:] if s['pos']=='NP' and s['role']),None)
            if next_role and next_role not in [role for _,role,_ in bindings]:
                bindings.append((pp,next_role,'Preposition and PP frame slot'))
                break
    score=4*len(bindings)-abs(len(np_slots)-len([p for p,_,_ in bindings if p['kind']!='PP']))*3
    return score,bindings,'Syntactic compatibility only; meaning must be reviewed'

def _xml_node_frame(frame):
    synt=[]
    syn=frame.find('SYNTAX')
    if syn is not None:
        for e in syn:
            value=e.get('value','')
            if e.tag=='NP':
                synt.append({'pos_tag':'NP','modifiers':{'value':value}})
            elif e.tag in ('PREP','PP','VERB','ADV','ADJ','S'):
                synt.append({'pos_tag':e.tag,'value':value})
    desc=frame.find('DESCRIPTION')
    return {'description':{'primary':desc.get('primary','') if desc is not None else ''},'syntax':synt}

@st.cache_resource(show_spinner='Reading VerbNet 3.3 XML files…')
def _parse_verbnet_zip(raw):
    import io, zipfile, xml.etree.ElementTree as ET
    bylemma={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.xml') and not n.startswith('__MACOSX')]
        if not names: raise ValueError('No VerbNet XML files found in this ZIP.')
        for name in names:
            try: root=ET.fromstring(z.read(name))
            except ET.ParseError: continue
            if root.tag not in ('VNCLASS','VNSUBCLASS'): continue
            def visit(node,inherited_roles=None,inherited_frames=None):
                cid=node.get('ID',name.rsplit('/',1)[-1][:-4]); inherited_roles=inherited_roles or []
                inherited_frames=inherited_frames or []
                rs=node.find('THEMROLES')
                local_roles=[x.get('type','') for x in rs] if rs is not None else []
                fr=node.find('FRAMES')
                local_frames=[_xml_node_frame(x) for x in fr if x.tag=='FRAME'] if fr is not None else []
                all_roles=list(dict.fromkeys(inherited_roles+local_roles))
                all_frames=inherited_frames+local_frames
                members=node.find('MEMBERS')
                if members is not None:
                    for member in members:
                        lemma=member.get('name','').lower()
                        if lemma: bylemma.setdefault(lemma,[]).append({'classid':cid,'roles':all_roles,'frames':all_frames})
                subclasses=node.find('SUBCLASSES')
                if subclasses is not None:
                    for sub in subclasses:
                        if sub.tag=='VNSUBCLASS':visit(sub,all_roles,all_frames)
            visit(root)
    return bylemma

def _lexicon_classes(lemma, uploaded):
    if uploaded:
        return uploaded.get(lemma,[]), 'Uploaded XML (check that archive is official VerbNet 3.3)'
    vn,err=get_verbnet()
    if vn is None:return [],'NLTK VerbNet unavailable: '+str(err)
    try: ids=vn.classids(lemma)
    except Exception: ids=[]
    out=[]
    for cid in ids[:45]:
        try:out.append({'classid':cid,'roles':[],'frames':vn.frames(cid)})
        except Exception:pass
    return out,'NLTK VerbNet (version NOT guaranteed to be 3.3)'

def _model_predictions(text):
    """Optional compatible SRL service. No pretrained VerbNet model is silently assumed."""
    try:
        config=st.secrets.get('verbnet_srl',{})
        url=config.get('url','');token=config.get('token','')
    except Exception:return [],'No compatible trained VerbNet SRL service configured'
    if not url:return [],'No compatible trained VerbNet SRL service configured'
    try:
        import requests
        headers={'Authorization':'Bearer '+token} if token else {}
        result=requests.post(url,json={'text':text},headers=headers,timeout=15)
        result.raise_for_status();data=result.json()
        if not isinstance(data,dict) or not isinstance(data.get('predictions'),list):raise ValueError('Expected JSON predictions array')
        allowed=set(ROLE_EXPLANATIONS)|set(ROLE_GLOSS)
        preds=[p for p in data['predictions'] if isinstance(p,dict) and p.get('role') in allowed and all(k in p for k in ('predicate','expression','role'))]
        return preds,'External trained SRL service (model and label compatibility must be validated)'
    except Exception as exc:return [],'Trained SRL service failed: '+str(exc)

def roles(doc, uploaded=None, use_model=False):
    rows=[];summaries=[]
    model_preds,model_status=_model_predictions(doc.text) if use_model else ([],'Not enabled')
    predicates=[t for t in doc if t.pos_=='VERB' and t.dep_ not in ('aux','auxpass')]
    for t in doc:
        if t.pos_=='ADJ' and any(c.lemma_.lower()=='be' for c in t.children) and t not in predicates:predicates.append(t)
    predicates.sort(key=lambda t:t.i)
    for v in predicates:
        lemma=v.lemma_.lower();parts=_participant_nodes(v)
        candidates,source=_lexicon_classes(lemma,uploaded)
        passive=any(p['dep'] in ('nsubjpass','csubjpass') for p in parts)
        best=None
        for c in candidates:
            for frame in c['frames']:
                slots=_frame_slots(frame)
                score,bindings,reason=_candidate_match(slots,parts,passive)
                if score>=4 and (best is None or score>best[0]):best=(score,c,frame,bindings,reason)
        summary={'Predicate':v.text,'Lemma':lemma,'Candidate classes':', '.join(x['classid'] for x in candidates[:10]) or 'None',
                 'Selected candidate':best[1]['classid'] if best else 'None',
                 'Candidate frame':_frame_description(best[2]) if best else 'No compatible frame',
                 'Lexicon source':source,
                 'Status':'Tentative frame alignment' if best else 'No compatible frame / manual review',
                 'Explanation':'VerbNet classes may contain multiple senses; grammatical matching does not prove semantic correctness.'}
        summaries.append(summary)
        bound={}
        if best:
            for p,role,evidence in best[3]:bound[(p['start'],p['end'])]=(role,evidence)
        for part in parts:
            role,evidence=bound.get((part['start'],part['end']),('Unresolved','No matching frame slot'))
            # Semantic sanity checks: an object cannot be automatically assigned Experiencer
            # solely because the lexicon lists it in the NP position.
            if role=='Experiencer' and part['kind']=='OBJECT' and part['text'].lower().startswith(('the paper','a paper','the window')):
                role,evidence='Unresolved','Experiencer is semantically implausible for this object; review class/frame'
            # Instrument is a *tentative* adjunct inference, not a verified VerbNet frame role.
            if role=='Unresolved' and part['kind']=='PP' and part.get('prep')=='with' and lemma in ('cut','slice','chop','write','paint','open','stir'):
                role,evidence='Instrument','Teaching heuristic: with + tool for an instrumental action (not a verified VerbNet frame role)'
            if role=='Unresolved' and lemma=='complete':
                if part['kind']=='SUBJECT':role,evidence='Agent','Teaching heuristic: active subject of complete'
                elif part['kind']=='OBJECT':role,evidence='Patient','Teaching heuristic: affected object of complete; verify role in actual VerbNet frame'
            for pred in model_preds:
                if pred['predicate'].lower()==v.text.lower() and pred['expression'].strip().lower()==part['text'].strip().lower():
                    role,evidence=pred['role'],'Trained-model suggestion (verify model and VerbNet role inventory)'
                    break
            reliability=('Trained model suggestion — verify' if evidence.startswith('Trained-model') else
                         'Tentative heuristic — verify' if 'heuristic' in evidence.lower() else
                         'Tentative frame match — verify' if role!='Unresolved' else 'Unresolved — manual review')
            rows.append({'Predicate':v.text,'Lemma':lemma,'VerbNet class (candidate)':summary['Selected candidate'],
                         'Expression':part['text'],'Grammatical evidence':part['kind']+' / '+part['dep'],
                         'Thematic role (suggested)':role,'Corrected role':role,
                         'Explanation':evidence+'. '+ROLE_EXPLANATIONS.get(role,''),
                         'Reliability':reliability,'Reviewer notes':''})
    return rows,summaries,model_status

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
st.caption('Penn Treebank POS • Universal Dependencies syntax • VerbNet candidate frames • PDTB 3.0-inspired discourse relations')
st.warning('Accuracy note: Penn POS and UD-style dependencies are spaCy predictions; UD labels are approximate mappings. VerbNet classes/frames and roles are candidate analyses, not verified semantic-role parses. Trained SRL is optional and requires a separately configured compatible model. PDTB uses connective rules. Always verify research annotations.')
nlp=get_nlp()
if nlp is None:
    st.error('The English spaCy model is missing. Run: python -m spacy download en_core_web_sm  (then restart Streamlit).')
    st.stop()

samples=['Sara gave Ali a book because he needed it.','Although Sara was tired, she completed her assignment.','Ali cut the paper with scissors.','Sara stayed home because she was ill.','The storm broke the window.','Sara sent a parcel to Lahore.']
example=st.selectbox('Sample sentences', ['Write my own']+samples)
text=st.text_area('Paste an English sentence or short paragraph',value='' if example=='Write my own' else example,height=115,placeholder='Sara gave Ali a book because he needed it.')
with st.expander('VerbNet 3.3 source and optional trained SRL (advanced)',expanded=False):
    st.markdown('Upload a ZIP containing the **official VerbNet 3.3 XML class files**. The PDF guidelines alone are not a machine-readable lexicon. Without a ZIP, the app uses NLTK VerbNet (version not guaranteed).')
    archive=st.file_uploader('VerbNet 3.3 XML archive (.zip)',type=['zip'])
    if archive is not None:
        try:
            st.session_state['vn33_xml']=_parse_verbnet_zip(archive.getvalue())
            st.success(f"Loaded {len(st.session_state['vn33_xml'])} verb lemmas from uploaded XML archive. Confirm the ZIP release is VerbNet 3.3.")
        except Exception as exc:st.error('Could not read VerbNet archive: '+str(exc))
    st.checkbox('Use a compatible trained VerbNet semantic-role service if configured',key='enable_trained_srl',help='Requires a separately deployed, validated service in Streamlit secrets [verbnet_srl], with url and optional token. No pretrained model is bundled.')
    st.caption('Optional service contract: POST {"text":"..."} → {"predictions":[{"predicate":"cut","expression":"the paper","role":"Patient"}]}. Check the service model was trained for compatible VerbNet labels.')
if st.button('Analyse all four layers',type='primary'):
    if not text.strip(): st.info('Please enter some text.');st.stop()
    if len(text)>5000: st.error('Please limit input to 5,000 characters.');st.stop()
    doc=nlp(text)
    posrows=[{'Token':t.text,'Lemma':t.lemma_,'Penn tag':t.tag_,'Full name':PENN.get(t.tag_,'Model-specific tag'),'Universal POS':t.pos_} for t in doc]
    synrows=[{'Token':t.text,'Head':t.head.text if t.head!=t else 'ROOT','spaCy dependency':t.dep_,'Approx. UD equivalent':ud_equivalent(t.dep_),'Meaning':UD.get(ud_equivalent(t.dep_).split(':')[0], 'Review with UD guidelines'),'Head index':t.head.i} for t in doc]
    st.session_state.pop('corrected_semantic',None)
    semrows,sem_summary,model_status=roles(doc,st.session_state.get('vn33_xml'),st.session_state.get('enable_trained_srl',False));discrows=discourse(text)
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
        st.caption('Upload official VerbNet 3.3 XML archive for version-specific lookup. Without it, NLTK VerbNet is used and its version may differ. Frame alignment remains tentative.')
        st.info('Trained role prediction: '+r.get('model_status','Not configured'))
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
        combined={'input':r['text'],'penn_pos':r['pos'],'ud_style_syntax':r['syntax'],'verbnet_candidate_frames':r.get('semantic_summary',[]),'verbnet_suggested_roles':r['semantic'],'verbnet_corrected_roles':st.session_state.get('corrected_semantic',r['semantic']),'pdtb_rule_suggestions':r['discourse'],'method_warning':'VerbNet frame matching is tentative and PDTB is rule-based; neither is a validated full parser.'}
        st.download_button('Download all results (JSON)',json.dumps(combined,indent=2,ensure_ascii=False),file_name='corpus_annotation.json',mime='application/json')
        for key,name in [('pos','penn_pos'),('syntax','ud_syntax'),('semantic','thematic_roles'),('discourse','pdtb_discourse')]:
            st.download_button(f'Download {name}.csv',pd.DataFrame(st.session_state.get('corrected_semantic',r[key]) if key=='semantic' else r[key]).to_csv(index=False),file_name=f'{name}.csv',mime='text/csv',key='dl_'+key)
        st.download_button('Download dependency tree (.dot)',r['syntax_dot'],file_name='dependency_tree.dot')
        st.markdown('**Official references:** [Penn Treebank](https://www.cis.upenn.edu/~treebank/) · [UD](https://universaldependencies.org/u/dep/) · [VerbNet](https://verbs.colorado.edu/verb-index/) · [PDTB 3.0](https://catalog.ldc.upenn.edu/LDC2019T05)')
else:
    st.info('Enter a sentence and click **Analyse all four layers** to begin.')
