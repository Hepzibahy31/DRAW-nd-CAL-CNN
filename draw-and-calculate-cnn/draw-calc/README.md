# Draw & Calculate (CNN project)

Draw a sum like `23 + 45 x 2` with the mouse. A CNN reads every character (digits 0-9 and + - x / =),
then the app solves it with its own calculator (no eval) and shows the steps.

## Run
```
pip install -r requirements.txt
python backend/train.py      # optional, trained files are already included (about 1.5 minutes)
python backend/app.py        # open http://127.0.0.1:5000
```
Connect to the internet when you open the page (fonts and Chart.js load online).

## The CNN (backend/cnn.py, written from scratch in NumPy)
Input 1x28x28 -> Conv2D(8, 3x3) + ReLU -> MaxPool -> Conv2D(16, 3x3) + ReLU -> MaxPool
-> Flatten(400) -> Dense(64) + ReLU -> Dropout -> Dense(15) + Softmax. About 27,900 parameters.
Trained with cross-entropy loss, backpropagation and the Adam optimizer.

## How the project works
1. `make_dataset.py` draws 10,500 characters with a virtual pen (random tilt, slant, size, wobble, thickness, blur, noise).
2. `train.py` also adds 1,797 real handwritten digits (scikit-learn) and trains the CNN, with random shifts each epoch.
3. `preprocess.py` cuts the drawing into characters at the gaps, then crops and centers each one in 28x28.
4. `calculator.py` solves the expression with BODMAS, exact fractions and clear error messages.
5. `app.py` (Flask) serves the API and the page in `frontend/`.

## Page features
- Draw pad with undo and clear, auto-calculate, step-by-step solution
- Fix any wrongly read character from its menu, the answer updates
- "Inside the CNN": input, feature maps of both conv layers, probabilities
- Training results: loss and accuracy graphs, confusion matrix, accuracy per character, most confused pairs, training images
- "Test my own handwriting": the CNN asks for a character, checks itself and keeps your accuracy.
  Run `python backend/train.py --with-mine` to train on your own drawings too.

## Honest notes
- Most training characters are generated, so the score on generated test images (100%) is easy. The real-digit score (about 98%) is the honest one.
- Operators have no real handwriting data. Test them with the "My handwriting" page and report your own accuracy.
