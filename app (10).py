"""PDTB 3.0 classroom discourse annotator.
Run: streamlit run app.py
Educational rule-based baseline, NOT a trained PDTB 3.0 parser.
"""
import re
import json
import csv
import io
import html
import streamlit as st

st.set_page_config(page_title='PDTB 3.0 Discourse Annotator', page_icon='🔎', layout='wide')

# Sense labels and plain-language descriptions adapted for classroom use.
SENSES = {
 'Temporal': {
  'Synchronous':'Events happen together', 'Asynchronous.Precedence':'Arg1 happens before Arg2',
  'Asynchronous.Succession':'Arg2 happens before Arg1'},
 'Contingency': {
  'Cause.Reason':'Arg2 explains why Arg1 happened', 'Cause.Result':'Arg2 is the result of Arg1',
  'Cause.NegResult':'An expected outcome does not happen',
  'Cause+Belief.Reason+Belief':'Arg2 supports a belief',
  'Cause+Belief.Result+Belief':'Evidence leads to a belief',
  'Cause+SpeechAct.Reason+SpeechAct':'Reason for saying or asking something',
  'Cause+SpeechAct.Result+SpeechAct':'A speech act follows from a reason',
  'Condition.Arg1-as-cond':'Arg1 is the condition',
  'Condition.Arg2-as-cond':'Arg2 is the condition',
  'Condition+SpeechAct':'Condition on saying or asking something',
  'Negative-condition.Arg1-as-negCond':'Arg1 gives a negative condition',
  'Negative-condition.Arg2-as-negCond':'Arg2 gives a negative condition',
  'Negative-condition+SpeechAct':'Negative condition on a speech act',
  'Purpose.Arg1-as-goal':'Arg1 is the intended goal',
  'Purpose.Arg2-as-goal':'Arg2 is the intended goal'},
 'Comparison': {
  'Concession.Arg1-as-denier':'Arg1 denies an expectation',
  'Concession.Arg2-as-denier':'Arg2 denies an expectation',
  'Concession+SpeechAct':'Concession involving a speech act',
  'Contrast':'Highlights differences', 'Similarity':'Highlights similarities'},
 'Expansion': {
  'Conjunction':'Adds information', 'Disjunction':'Presents alternatives',
  'Equivalence':'Restates the same idea',
  'Exception.Arg1-as-excpt':'Arg1 gives an exception',
  'Exception.Arg2-as-excpt':'Arg2 gives an exception',
  'Instantiation.Arg1-as-instance':'Arg1 provides an example',
  'Instantiation.Arg2-as-instance':'Arg2 provides an example',
  'Level-of-detail.Arg1-as-detail':'Arg1 provides detail',
  'Level-of-detail.Arg2-as-detail':'Arg2 provides detail',
  'Manner.Arg1-as-manner':'Arg1 shows how',
  'Manner.Arg2-as-manner':'Arg2 shows how',
  'Substitution.Arg1-as-subst':'Arg1 is the replacement',
  'Substitution.Arg2-as-subst':'Arg2 is the replacement'}
}
RELATION_TYPES=['Explicit','Implicit','AltLex','AltLexC','EntRel','Hypophora','NoRel']
# Mapping is deliberately small and conservative: words like 'since', 'while', 'so' are ambiguous.
RULES = [
 ('because','Contingency.Cause.Reason','subordinate'),
 ('although','Comparison.Concession.Arg1-as-denier','subordinate'),
 ('even though','Comparison.Concession.Arg1-as-denier','subordinate'),
 ('though','Comparison.Concession.Arg1-as-denier','subordinate'),
 ('if','Contingency.Condition.Arg2-as-cond','subordinate'),
 ('unless','Contingency.Negative-condition.Arg2-as-negCond','subordinate'),
 ('before','Temporal.Asynchronous.Precedence','subordinate'),
 ('after','Temporal.Asynchronous.Succession','subordinate'),
 ('when','Temporal.Synchronous','subordinate'),
 ('while','Temporal.Synchronous','subordinate'),
 ('whereas','Comparison.Contrast','subordinate'),
 ('because of','Contingency.Cause.Reason','subordinate'),
 ('however','Comparison.Contrast','adverbial'),
 ('therefore','Contingency.Cause.Result','adverbial'),
 ('consequently','Contingency.Cause.Result','adverbial'),
 ('moreover','Expansion.Conjunction','adverbial'),
 ('furthermore','Expansion.Conjunction','adverbial'),
 ('similarly','Comparison.Similarity','adverbial'),
 ('for example','Expansion.Instantiation.Arg2-as-instance','adverbial'),
 ('in other words','Expansion.Equivalence','adverbial'),
 ('instead','Expansion.Substitution.Arg2-as-subst','adverbial'),
 ('but','Comparison.Contrast','coordinator'),
 ('and','Expansion.Conjunction','coordinator'),
 ('or','Expansion.Disjunction','coordinator'),
]
RULES.sort(key=lambda x:len(x[0]), reverse=True)
EXAMPLES=[
 'Sara stayed home because she was ill.',
 'Because she was ill, Sara stayed home.',
 'Although Sara was tired, she completed her assignment.',
 'If it rains, we will stay inside.',
 'She left before he arrived.',
 'Ali likes tea, whereas Sara likes coffee.',
 'She teaches; moreover, she researches.',
 'It rained. The roads flooded.'
]

def tidy(s):
 return re.sub(r'\s+', ' ', s).strip(' ,;:.!?\n\t')

def level_parts(sense):
 major, _, minor=sense.partition('.')
 return major, minor

def extract(text):
 """Return one provisional explicit relation or an abstention; no invented implicit classification."""
 original=text.strip()
 if not original:
  return None
 for conn,sense,kind in RULES:
  match=re.search(r'(?<!\w)'+re.escape(conn)+r'(?!\w)',original,flags=re.I)
  if not match: continue
  before=original[:match.start()]
  after=original[match.end():]
  # Preposed subordinate clause: Because X, Y.
  if kind=='subordinate' and not tidy(before):
   sep=re.search(r'[,;]',after)
   if not sep: continue
   a2=tidy(after[:sep.start()]); a1=tidy(after[sep.end():])
  elif kind=='subordinate':
   a1=tidy(before); a2=tidy(after)
  elif kind=='coordinator':
   a1=tidy(before); a2=tidy(after)
  else:
   # Discourse adverbial after punctuation, e.g. X; therefore, Y
   a1=tidy(before); a2=tidy(after)
  if not a1 or not a2: continue
  notes=[]
  if conn in ('while','when','before','after','if','but','and','or'):
   notes.append('This connective has multiple possible meanings; verify the sense in context.')
  if conn in ('although','though','even though'):
   notes.append('Concession direction requires interpretation; check the denier argument manually.')
  if conn=='because of':
   notes.append('A prepositional phrase may not be a full discourse argument; verify applicability.')
  if re.search(r'[.!?]\s+\S',original):
   notes.append('Multiple sentences detected; only one candidate relation is shown.')
  return dict(arg1=a1,arg2=a2,connective=original[match.start():match.end()],
              relation_type='Explicit',sense=sense,confidence='Low — heuristic',
              notes=notes or ['Provisional automatic suggestion: review Arg1, Arg2 and sense.'])
 return dict(arg1='',arg2='',connective='',relation_type='Unresolved',sense='',
             confidence='Not classified',notes=['No supported explicit connective pattern found. This does not mean NoRel: implicit, AltLex and other relations require review.'])

def tree_dot(r):
 def q(s): return json.dumps(str(s),ensure_ascii=False)
 major,minor=level_parts(r['sense']) if r['sense'] else ('Unresolved','')
 dot=['digraph D {','graph [rankdir=TB, bgcolor="transparent", pad="0.3"];',
      'node [shape=box, style="rounded,filled", fillcolor="#edf3fa", color="#94a3b8", fontname="Arial", fontsize=12];',
      'edge [color="#64748b"];',
      'root [label="Discourse relation"];',
      'kind [label='+q('Type: '+r['relation_type'])+'];',
      'sense [label='+q('Sense: '+(r['sense'] or 'Unresolved'))+'];',
      'a1 [label='+q('Arg1: '+(r['arg1'] or 'Not identified'))+'];',
      'a2 [label='+q('Arg2: '+(r['arg2'] or 'Not identified'))+'];',
      'conn [label='+q('Connective: '+(r['connective'] or 'None detected'))+'];',
      'root -> kind; root -> sense; root -> conn; root -> a1; root -> a2;', '}']
 return '\n'.join(dot)

st.title('PDTB 3.0 | Discourse Annotation')
st.caption('Teaching prototype • Rule-based explicit-relation suggestions • Not a validated PDTB 3.0 automatic parser')
st.warning('This tool does NOT automatically recognize the full PDTB 3.0 hierarchy. It suggests one explicit relation using simple patterns, and lets you correct it. It does not reliably classify implicit relations or complex paragraphs.')
with st.sidebar:
 st.header('Quick guide')
 st.markdown('**Arg1:** one connected text span.  \n**Arg2:** the other span, often the clause introduced by *because / although / if*.  \n**Connective:** linking word or phrase.  \n**Sense:** meaning of the relationship.')
 st.markdown('**3 steps:** Find the connective → find the two spans → check the sense.')
 st.markdown('[PDTB 3.0 annotation manual](https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf)')
 st.divider()
 st.caption('All sense labels below are included as manual correction options, not as automatic model capabilities.')
 with st.expander('Complete sense inventory'):
  for category,senses in SENSES.items():
   st.markdown('**'+category+'**')
   for label,desc in senses.items(): st.caption(label+' — '+desc)
 with st.expander('Relation types'):
  for typ in RELATION_TYPES: st.write('• '+typ)

preset=st.selectbox('Try an example (optional)', ['Custom sentence']+EXAMPLES)
input_text=st.text_area('Paste one sentence or a short text', value=EXAMPLES[0] if preset=='Custom sentence' else preset, height=105, key='sentence')
if st.button('Analyse discourse',type='primary',use_container_width=True):
 st.session_state['analysis']=extract(input_text)
 st.session_state['source']=input_text
if 'analysis' in st.session_state:
 r=st.session_state['analysis']
 st.subheader('1. Automatic suggestion')
 st.caption('Source text: '+st.session_state.get('source',''))
 if r['relation_type']=='Unresolved': st.info('No reliable automatic annotation produced. You can complete the annotation manually below.')
 else: st.success('A candidate explicit relation was detected. Review before using it as a final annotation.')
 col1,col2=st.columns(2)
 with col1:
  st.markdown('**Arg1**');st.info(r['arg1'] or 'Not identified')
  st.markdown('**Connective**');st.info(r['connective'] or 'Not identified')
 with col2:
  st.markdown('**Arg2**');st.info(r['arg2'] or 'Not identified')
  st.markdown('**Proposed sense**');st.info(r['sense'] or 'Unresolved')
 st.write('**Relation type:**',r['relation_type'],'  |  **Confidence:**',r['confidence'])
 for n in r['notes']: st.caption('⚠ '+n)
 st.subheader('2. Check or correct the annotation')
 st.caption('Corrections are manual; the software does not learn from them automatically.')
 with st.form('corrections'):
  a1=st.text_area('Arg1',value=r['arg1'],height=75)
  a2=st.text_area('Arg2',value=r['arg2'],height=75)
  conn=st.text_input('Connective (leave blank if absent)',value=r['connective'])
  types=['Unresolved']+RELATION_TYPES
  reltype=st.selectbox('Relation type',types,index=types.index(r['relation_type']) if r['relation_type'] in types else 0)
  senses=['Unresolved']+[f'{cat}.{key}' for cat, entries in SENSES.items() for key in entries]
  sense=st.selectbox('PDTB 3.0 sense',senses,index=senses.index(r['sense']) if r['sense'] in senses else 0)
  note=st.text_area('Researcher notes',value='')
  submit=st.form_submit_button('Apply corrections',type='primary')
 if submit:
  st.session_state['corrected']=dict(arg1=a1,arg2=a2,connective=conn,relation_type=reltype,sense='' if sense=='Unresolved' else sense,confidence='Manually reviewed',notes=[note] if note else [])
  st.session_state['corrected_source']=st.session_state['source']
 corrected=st.session_state.get('corrected') if st.session_state.get('corrected_source')==st.session_state.get('source') else None
 shown=corrected or r
 st.subheader('3. Visual discourse tree')
 st.graphviz_chart(tree_dot(shown),use_container_width=True)
 st.caption('This is a relationship diagram (not a syntactic parse tree).')
 st.subheader('4. Detailed annotation table')
 major,minor=level_parts(shown['sense']) if shown['sense'] else ('Unresolved','Unresolved')
 rows=[('Original text',st.session_state['source']),('Arg1',shown['arg1']),('Arg2',shown['arg2']),('Connective',shown['connective'] or 'None'),('Relation type',shown['relation_type']),('Level 1',major),('Level 2 / 3',minor),('Full sense',shown['sense'] or 'Unresolved'),('Status',shown['confidence'])]
 for label,value in rows: st.markdown(f'**{label}:** {html.escape(str(value))}')
 st.subheader('5. Export')
 record={k:v for k,v in rows}
 record['Researcher notes']='; '.join(shown['notes'])
 st.download_button('Download annotation as JSON',data=json.dumps(record,indent=2,ensure_ascii=False),file_name='pdtb_annotation.json',mime='application/json')
 buff=io.StringIO();writer=csv.DictWriter(buff,fieldnames=list(record));writer.writeheader();writer.writerow(record)
 st.download_button('Download annotation as CSV (Excel-compatible)',data=buff.getvalue().encode('utf-8-sig'),file_name='pdtb_annotation.csv',mime='text/csv')
 st.download_button('Download visual tree (Graphviz DOT)',data=tree_dot(shown),file_name='pdtb_tree.dot',mime='text/vnd.graphviz')
 st.caption('For formal research, independently verify every annotation against the PDTB manual and document inter-annotator agreement.')
