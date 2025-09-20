1. Install the Required Libraries
Run the following command in your terminal to install the required Python libraries:
bash pip install torch torchvision flask pillow pandas scikit-learn matplotlib seaborn tqdm

 2. Verify Installation
To check that the libraries are installed correctly, you can run:
bash python -c "import torch; import torchvision; import flask; import PIL; import pandas; import sklearn; import matplotlib; import seaborn; import tqdm; print('All libraries installed successfully!')"

3. Run the Flask App
Ensure the `app.py` file contains your Flask application code.
1. Open the terminal in the directory where `app.py` is located.
2. Start the Flask app using the command:
bash python app.py

3. If successful, you will see something like:
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

4. Access the Flask App
- Open a web browser.
- Visit: `http://127.0.0.1:5000/`