  
const TRAIN_TEST_RATIO = 5 / 6;
const LEARNING_RATE = .05;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 100;
const IMAGE_SIZE = 784;
const LABELS_SIZE = 10;


const myMath = dl.ENV.math;

var myData;

const weights = dl.variable(dl.Array2D.randNormal([IMAGE_SIZE, LABELS_SIZE], 0, 1 / Math.sqrt(IMAGE_SIZE), 'float32'));

const optimizer = new dl.SGDOptimizer(LEARNING_RATE);


const mnistConfig = {
    'data': [
        {
            'name': 'images',
            'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
                'mnist_images.png',
            'dataType': 'png',
            'shape': [28, 28, 1]
        },
        {
            'name': 'labels',
            'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
                'mnist_labels_uint8',
            'dataType': 'uint8',
            'shape': [10]
        }
    ],
    modelConfigs: {}
};



///////////////////////// End Initilizations //////////////////////////////////////////////////

// Main class that is used
class MnistData {   
    
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }
    
    
    nextTrainBatch(batchSize) {
        return this.nextBatch(batchSize, this.trainingData, () => {
            this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
            return this.trainIndices[this.shuffledTrainIndex];
        });
    }
    
    
    
    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, this.testData, () => {
            this.shuffledTestIndex =
                (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        });
    }
    
    
    
    nextBatch(batchSize, myData, index) {
        let xs = null;
        let labels = null;
        for (let i = 0; i < batchSize; i++) {
            const idx = index();
            const x = myData[0][idx].reshape([1, 784]);
            xs = concatWithNulls(xs, x);
            const label = myData[1][idx].reshape([1, 10]);
            labels = concatWithNulls(labels, label);
        }
        return { xs, labels };
    }
   
    // Load all the MNIST data 
    async load() {
        this.dataset = new dl.XhrDataset(mnistConfig);
        await this.dataset.fetchData();
        this.dataset.normalizeWithinBounds(0, -1, 1);
        this.trainingData = this.getTrainingData();
        this.testData     = this.getTestData();
        this.trainIndices = dl.util.createShuffledIndices(this.trainingData[0].length);
        this.testIndices  = dl.util.createShuffledIndices(this.testData[0].length);
    }
    
    
    getTrainingData() {
        const [images, labels] = this.dataset.getData();
        const end = Math.floor(TRAIN_TEST_RATIO * images.length);
        return [images.slice(0, end), labels.slice(0, end)];
    }
    
    
    getTestData() {
        const myData = this.dataset.getData();
        if (myData == null) { return null; }
        const [images, labels] = this.dataset.getData();
        const start = Math.floor(TRAIN_TEST_RATIO * images.length);
        return [images.slice(start), labels.slice(start)];
    }
    
}  // end of class MnistData  definition


// A helper for the NextBatch function above
function concatWithNulls(ndarray1, ndarray2) {
    if (ndarray1 == null && ndarray2 == null) {
        return null;
    }
    if (ndarray1 == null) {
        return ndarray2;
    }
    else if (ndarray2 === null) {
        return ndarray1;
    }
    return myMath.concat2D(ndarray1, ndarray2, 0);
}


 
/////////////////////////// End of the Class Definition and Helper function ///////////////


 


// Train the model.
async function train(myData) {
    const returnCost = true;
    for (let i = 0; i < TRAIN_STEPS; i++) {
        const cost = optimizer.minimize(() => {
            const batch = myData.nextTrainBatch(BATCH_SIZE);
            return myMath.mean(myMath.softmaxCrossEntropyWithLogits(batch.labels, myMath.matMul(batch.xs, weights)));
        }, returnCost);
        document.getElementById('message').innerHTML = 'loss[' + i + ']:' +  cost.dataSync()
        await dl.util.nextFrame();
    }
}





/////////////////////////////////// End Training the Machine /////////////////////////////////////////////////////////////




// Predict the digit number from a batch of input images.
function predict(x) {
    const pred = myMath.scope(() => {
        const axis = 1;
        return myMath.argMax(myMath.matMul(x, weights), axis);
    });
    return Array.from(pred.dataSync());
}


async function showTestResults(batch, predictions, labels) {
    var testExamples = batch.xs.shape[0];
    var totalCorrect = 0;
    for (var i = 0; i < testExamples; i++) {
        var image = myMath.slice2D(batch.xs, [i, 0], [1, batch.xs.shape[1]]);
        var div = document.createElement('div');
        div.className = 'pred-container';
        var canvas = document.createElement('canvas');
        draw(image.flatten(), canvas);
        var pred = document.createElement('div');
        var prediction = predictions[i];
        var label = labels[i];
        var correct = prediction === label;
        if (correct) {
            totalCorrect++;
        }
        pred.className = "pred " + (correct ? 'pred-correct' : 'pred-incorrect');
        pred.innerText = "pred: " + prediction;
        
        div.appendChild(pred);
        div.appendChild(canvas);
        document.getElementById('images').appendChild(div);
    }
    var accuracy = 100 * totalCorrect / testExamples;
    var displayStr = 'Accuracy: ' + accuracy.toFixed(2) + ' % (' + totalCorrect + ' / ' + testExamples + ')';
    document.getElementById('message').innerHTML += '<br>'+ displayStr + '<br>';
    //console.log(displayStr);
}



function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const myData = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = myData[i] * 255;
        imageData.data[j + 1] = myData[i] * 255;
        imageData.data[j + 2] = myData[i] * 255;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

/////////////////////////// End Drawing the Images //////////////////////////////////////


// Main Entry point
async function myMnist() {
    
    // Load Mnist Dataset
    document.getElementById('status').innerText = ' Loading...'
    myData = new MnistData(); 
    await myData.load();      
    
    // Train the network
    document.getElementById('status').innerText += ' Training...'
    await train(myData);
    
    // Make some predictions, testing the network
    
    document.getElementById('images').innerText = ''  
    document.getElementById('status').innerText += ' Testing...';
    const testExamples = 50;
    const batch = myData.nextTestBatch(testExamples);
    const predictions = predict(batch.xs);

    // Given a logits or label vector, return the class indices.
    const axis = 1;
    const pred = myMath.argMax(batch.labels, axis);
    const labels = Array.from(pred.dataSync());
    
    
    document.getElementById('status').innerText += ' Predicting...';
    showTestResults(batch, predictions, labels);
}  // End of Main program


/////////////////////// End main Program /////////////////////////////////
















