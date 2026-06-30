# Deployment Notes

A Render deployment was configured (`render.yaml` in repo root, deploy hook in CI). The app builds and starts successfully, but the free-tier 512 MB RAM limit is insufficient to load BGE-small embeddings + chromadb + Flask simultaneously, causing OOM restarts on first request. 

**Live URL:** https://rag-policies-oc44.onrender.com

**For grading:** please run the app locally using the README instructions. 
A demo recording of the working local app is linked in the submission PDF.

A paid Render Starter tier (2 GB RAM) or Hugging Face Spaces would resolve this. The deployment configuration is preserved in the repo as evidence of a working CI/CD pipeline through to deployment.


