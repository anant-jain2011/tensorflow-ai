import tf from '@tensorflow/tfjs';
import use from '@tensorflow-models/universal-sentence-encoder';

async function getEmbeddings(sentences) {
  const useModel = await use.load();
  return await useModel.embed(sentences);
}

async function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [512], units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 2, activation: 'softmax' })); // binary classification

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

async function train() {
  const model = await createModel();

  // Example data: 1 = positive, 0 = negative
  const sentences = [
    'I love this product',
    'This is terrible',
    'Amazing experience',
    'Worst ever',
    'I would buy this again',
    'Not worth the money',
  ];
  const labels = [1, 0, 1, 0, 1, 0];

  const embeddings = await getEmbeddings(sentences);
  const ys = tf.tensor2d(labels.map(label => label === 1 ? [0, 1] : [1, 0]));

  await model.fit(embeddings, ys, {
    epochs: 50,
    batchSize: 2,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} accuracy=${logs.acc.toFixed(4)}`)
    }
  });

  return model;
}

async function predict(model, sentence) {
  const embedding = await getEmbeddings([sentence]);
  const prediction = model.predict(embedding);
  const result = await prediction.data();
  console.log(`Prediction for "${sentence}":`);
  console.log(`Negative: ${result[0].toFixed(4)}, Positive: ${result[1].toFixed(4)}`);
}

(async () => {
  const model = await train();
  await predict(model, 'I really enjoyed this movie!');
  await predict(model, 'This is the worst thing I\'ve ever bought.');
})();

// predict(model, 'I really enjoyed this movie!');