"""Four-layer classroom annotation demonstrator. Run: streamlit run app.py"""
import re
import json
import pandas as pd
import streamlit as st
import spacy
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

def roles(doc):
    results=[]
    for v in doc:
        if v.pos_ != 'VERB' or v.lemma_.lower() not in LOOKUP: continue
        group, subjrole, objrole, extra = LOOKUP[v.lemma_.lower()]
        passive = any(c.dep_ in ('auxpass','aux:pass') for c in v.children)
        for c in v.children:
            dep=c.dep_.split(':')[0]
            role=None
            if dep in ('nsubj','csubj'):
                role = objrole if passive and objrole else subjrole
            elif dep in ('dobj','obj','attr','oprd'):
                role = objrole
            elif dep=='iobj':
                role='Recipient'
            elif dep in ('prep','agent','obl'):
                p=c.text.lower()
                if p in ('with','using'): role='Instrument' if group in ('Cutting','Cooking') else 'Co-Agent/Co-Theme'
                elif p in ('to','into','onto','toward','towards'): role='Recipient' if group in ('Giving','Selling','Communication') else 'Destination'
                elif p in ('from','out','of'): role='Source' if group in ('Removing','Buying') else 'Initial_Location'
                elif p in ('through','across','along'): role='Path'
                elif p in ('for',): role='Beneficiary' if group=='Helping' else ('Asset' if group=='Buying' else 'Purpose')
                elif p in ('at',): role='Value' if group=='Measuring' else 'Location'
                elif p in ('by',) and passive: role='Agent'
                elif p in ('in','on','under'): role='Location'
                if role and c.pos_=='ADP':
                    complement=next((x for x in c.children if x.dep_ in ('pobj','obj')),None)
                    if complement is not None:
                        results.append({'Verb':v.text,'Verb group':group,'Expression':phrase(complement),'Thematic role':role,'Evidence':'Prepositional phrase (heuristic)','Status':'Suggested; verify class/frame'})
                    continue
            if role:
                results.append({'Verb':v.text,'Verb group':group,'Expression':phrase(c),'Thematic role':role,'Evidence':c.dep_,'Status':'Suggested; verify class/frame'})
        if not any(r['Verb']==v.text for r in results):
            results.append({'Verb':v.text,'Verb group':group,'Expression':'—','Thematic role':'Not resolved','Evidence':'No matching dependency','Status':'Manual annotation required'})
    return results

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
    verbs=list(dict.fromkeys(r['Verb'] for r in rows))
    for i,v in enumerate(verbs):
        lines.append(f'v{i} [label={json.dumps("VERB: "+v)},fillcolor="#ffe9cc"];')
        for j,r in enumerate(x for x in rows if x['Verb']==v):
            key=f'r{i}_{j}'
            lines.append(f'{key} [label={json.dumps(r["Expression"]+"\n"+r["Thematic role"])}];')
            lines.append(f'v{i} -> {key} [label={json.dumps(r["Thematic role"])}];')
    return '\n'.join(lines+['}'])

UD_NORMALIZE={'ROOT':'root','dobj':'obj','pobj':'obl','prep':'case','nsubjpass':'nsubj:pass','csubjpass':'csubj:pass','auxpass':'aux:pass','poss':'nmod:poss','npadvmod':'obl','acomp':'xcomp','attr':'xcomp','relcl':'acl:relcl','agent':'obl:agent','prt':'compound:prt','oprd':'xcomp','neg':'advmod','dative':'iobj','predet':'det:predet'}
def ud_equivalent(label):
    return UD_NORMALIZE.get(label,label if label in UD else 'review manually')

def dot_disc(row):
    return 'digraph D {rankdir=TB; node [shape=box,style="rounded,filled",fillcolor="#eef5ff"]; root [label='+json.dumps(row['Sense suggestion'])+']; root -> arg1; root -> conn; root -> arg2; arg1 [label='+json.dumps('Arg1: '+row['Arg1'])+']; conn [label='+json.dumps('Connective: '+row['Connective'])+']; arg2 [label='+json.dumps('Arg2: '+row['Arg2'])+'];}'

st.title('📚 LinguAnnotate 4D — Four-Layer Annotation Studio')
st.caption('Penn Treebank POS • Universal Dependencies syntax • VerbNet-inspired thematic roles • PDTB 3.0-inspired discourse relations')
st.warning('Accuracy note: Penn POS and UD dependencies are model predictions. VerbNet and PDTB sections are transparent classroom heuristics, NOT validated full VerbNet 3.3 / PDTB 3.0 automatic parsers. Check against your uploaded cheat sheets and official manuals.')
nlp=get_nlp()
if nlp is None:
    st.error('The English spaCy model is missing. Run: python -m spacy download en_core_web_sm  (then restart Streamlit).')
    st.stop()

samples=['Sara gave Ali a book because he needed it.','Although Sara was tired, she completed her assignment.','Ali cut the paper with scissors.','Sara stayed home because she was ill.','The storm broke the window.','Sara sent a parcel to Lahore.']
example=st.selectbox('Sample sentences', ['Write my own']+samples)
text=st.text_area('Paste an English sentence or short paragraph',value='' if example=='Write my own' else example,height=115,placeholder='Sara gave Ali a book because he needed it.')
if st.button('Analyse all four layers',type='primary'):
    if not text.strip(): st.info('Please enter some text.');st.stop()
    if len(text)>5000: st.error('Please limit input to 5,000 characters.');st.stop()
    doc=nlp(text)
    posrows=[{'Token':t.text,'Lemma':t.lemma_,'Penn tag':t.tag_,'Full name':PENN.get(t.tag_,'Model-specific tag'),'Universal POS':t.pos_} for t in doc]
    synrows=[{'Token':t.text,'Head':t.head.text if t.head!=t else 'ROOT','spaCy dependency':t.dep_,'Approx. UD equivalent':ud_equivalent(t.dep_),'Meaning':UD.get(ud_equivalent(t.dep_).split(':')[0], 'Review with UD guidelines'),'Head index':t.head.i} for t in doc]
    semrows=roles(doc);discrows=discourse(text)
    st.session_state['results']={'pos':posrows,'syntax':synrows,'semantic':semrows,'discourse':discrows,'syntax_dot':dot_syntax(doc),'text':text}

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
        st.subheader('VerbNet-inspired thematic roles (full names)')
        if r['semantic']:
            st.markdown('**Visual semantic-role map** — arrows connect a verb to its suggested participants and full role names.')
            st.graphviz_chart(dot_roles(r['semantic']),use_container_width=True)
            st.dataframe(pd.DataFrame(r['semantic']),use_container_width=True,hide_index=True)
            st.markdown('**How to interpret:** Verb group indicates a teaching category; Expression is the participant phrase; Thematic role is its proposed meaning in the event; Evidence shows the grammatical clue. **This is not a verified VerbNet 3.3 class/frame assignment.**')
            st.info('These are suggested participant roles from the 20 verb groups in your cheat sheet. They are NOT predictions of official VerbNet classes, senses, or syntactic frames. Some roles may be wrong or absent.')
        else: st.info('No supported verb-group match. Use the reference table and annotate manually.')
        with st.expander('Role definitions'):
            st.dataframe(pd.DataFrame([{'Role':k,'Simple meaning':v} for k,v in ROLE_GLOSS.items()]),hide_index=True,use_container_width=True)
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
        combined={'input':r['text'],'penn_pos':r['pos'],'ud_style_syntax':r['syntax'],'verbnet_inspired_roles':r['semantic'],'pdtb_rule_suggestions':r['discourse'],'method_warning':'VerbNet and PDTB results are heuristic, not validated full parsers.'}
        st.download_button('Download all results (JSON)',json.dumps(combined,indent=2,ensure_ascii=False),file_name='corpus_annotation.json',mime='application/json')
        for key,name in [('pos','penn_pos'),('syntax','ud_syntax'),('semantic','thematic_roles'),('discourse','pdtb_discourse')]:
            st.download_button(f'Download {name}.csv',pd.DataFrame(r[key]).to_csv(index=False),file_name=f'{name}.csv',mime='text/csv',key='dl_'+key)
        st.download_button('Download dependency tree (.dot)',r['syntax_dot'],file_name='dependency_tree.dot')
        st.markdown('**Official references:** [Penn Treebank](https://www.cis.upenn.edu/~treebank/) · [UD](https://universaldependencies.org/u/dep/) · [VerbNet](https://verbs.colorado.edu/verb-index/) · [PDTB 3.0](https://catalog.ldc.upenn.edu/LDC2019T05)')
else:
    st.info('Enter a sentence and click **Analyse all four layers** to begin.')
