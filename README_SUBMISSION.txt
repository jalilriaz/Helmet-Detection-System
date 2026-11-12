================================================================================
CS619 Final Year Project - Final Deliverable
Project: Automated Motorcycle Helmet Detection System
================================================================================

CONTENTS OF THIS SUBMISSION:
1. Final Project Report: Helmet_Detection_Final_Project_Report (112).pdf
2. Final Viva Presentation: Helmet_Detection_Viva_Presentation_Updated.pptx
3. Complete Application Code: app.py, requirements.txt, packages.txt, data.yaml
4. Training Notebooks:
   - HELMETY8_v4.ipynb
   - HELMETY8_v5_EdgeVision.ipynb
   - Helmet_Detection_using_Yolo11.ipynb
   - Helmet_Detection_YOLOv11.ipynb
5. Test Sample Images: images/
6. Lightweight Model Weights: yolov11nbest.pt (Included in this package)

--------------------------------------------------------------------------------
NOTE REGARDING HEAVY MODEL WEIGHTS (best.pt - 50 MB):
--------------------------------------------------------------------------------
Due to the VULMS submission limit of 25 MB, the heavy YOLOv8 Medium trained 
model weights (best.pt, 50 MB) have been shared with the supervisor via Google Drive 
using the official VU Student Email as instructed in the CS619 announcement.

Google Drive Link (Full Backup with best.pt):
[INSERT_YOUR_GOOGLE_DRIVE_SHARE_LINK_HERE]

--------------------------------------------------------------------------------
HOW TO RUN THE APPLICATION:
--------------------------------------------------------------------------------
1. Install required packages:
   pip install -r requirements.txt

2. Run the application (Gradio Web Interface):
   python app.py

3. Access in browser at:
   http://127.0.0.1:7860

4. Run direct high-speed desktop webcam:
   python app.py --webcam
================================================================================

