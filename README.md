# Hybrid Multiple-Pit Inference 

This repo organizes the analysis into **five** clean steps:

1. **Extract & analyze measurements** (`step1_measurements.py`)  
   - Input: `data/si_area_values.xlsx` (private)  
   - Output: 5×1000 KDE curves (not saved by default)

2. **GPR surrogate (single pit)** (`step2_surrogate.py`)  
   - Loads `checkpoints/GPexperiment.dump` (private)

3. **Hybrid simulation helpers** (`step3_hybrid.py`)  
   - `number_of_pits_curve(...)` for cumulative counts  
   - `simulate_depth_pdf(...)` for PDFs via the GP surrogate

4. **CI-NN model updating** (`step4_update.py`)  
   - Inputs: `data/input_beta.pkl`, `data/output_beta.pkl`, `checkpoints/weights_beta` (private)  
   - Output: `data/post_samples.csv` (posterior samples)

5. **Predict with updated model** (`step5_predict.py`)  
   - Uses posterior mean(s) + GP to produce:  
     - `data/cumulative_pits.csv`  
     - `data/depth_pdfs.csv`


