<h2>Abstract</h2>
<par> Glaucoma is the second leading cause of loss of vision in the world. Examining the head of optic nerve (cup-to-disc ratio) among other various metrics is crucial for diagnosing glaucoma. </par>

<h2>Introduction</h2>
<par> This code adopts a two-step approach for detecting glaucoma: Initially, AI identifies the contours of the optic cup and disc, then uses this data to derive an automated decision rule for glaucoma detection in the second step. </par>

<br><br>
<h2>First Step: Segmentation</h2>

![unet-proc](https://github.com/vladradu21/2024-Glaucoma/assets/117584846/6baaf580-3f22-4eb7-9537-c68a489f50df)

<par> The initial step employs a UNET architecture with three output channels, processing the fundus eye image to segment the optic cup (black) and the optic disc (gray) to generate the mask. </par>

<br>
<h2>Second Step: Classification</h2>
<par> This step unfolds in two distinct phases: </par>

<h3>Phase One: Metric Calculation</h3>

![glaucoma-metrics](https://github.com/vladradu21/2024-Glaucoma/assets/117584846/69804464-34d2-4b99-8cf1-afe3bc4bfe4a)

<par> Metrics such as the cup to disc ratio (CDR), horizontal/vertical cup to disc diameter (hCDR/vCDR), and neuroretinal rim area ratio (NRR) based on the ISNT areas are computed from the ROI and recorded in a CSV file. </par>

<h3>Phase Two: Classification</h3>

<par> A classification model is defined and trained using the metrics and ground truth (diagnosis) from fundus eye images evaluated by expert medics. This model is then used to make predictions. </par>

<br><br>
<h2>Setup</h2>
