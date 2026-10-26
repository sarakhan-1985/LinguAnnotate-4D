LinguAnnotate 4D — Streamlit deployment

Upload all three files into the same GitHub repository folder:
  app.py
  requirements.txt
  verbnet_data.zip

In Streamlit Cloud set Main file path to: app.py
Do not leave Main file path as app (11).py if you uploaded app.py.
Then reboot or redeploy the app.

The ZIP is read automatically from the directory containing app.py.
No manual upload is required for students.

IMPORTANT: This is a complete runnable four-layer educational prototype.
It uses spaCy predictions, VerbNet XML frame matching, and connective-based
PDTB suggestions. It does NOT run the original VerbNetParser.com pretrained
semantic parser. That requires its separate model/runtime deployment.
