import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { peekLayers } from '../dist/index.js';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const FIXTURE = resolve(dirname(fileURLToPath(import.meta.url)), '../example/model-3pl/model.json');

describe('peekLayers', () => {
  test('model metadata', async () => {
    const { model } = await peekLayers(FIXTURE);
    assert.equal(model.name, 'football-predict');
    assert.equal(model.totalParams, 97);
    assert.deepEqual(model.inputShape, [null, 8]);
    assert.deepEqual(model.outputShape, [null, 3]);
    assert.equal(model.trainableWeights, 6);
    assert.equal(model.nonTrainableWeights, 0);
  });

  test('returns 3 layers', async () => {
    const { layers } = await peekLayers(FIXTURE);
    assert.equal(layers.length, 3);
  });

  test('layer has shape, config and stats', async () => {
    const { layers } = await peekLayers(FIXTURE);
    const layer = layers[0];
    assert.equal(layer.name, 'dense_Dense1');
    assert.deepEqual(layer.weights.shape, [8, 6]);
    assert.deepEqual(layer.bias.shape, [6]);
    assert.equal(typeof layer.weights.stats.min, 'number');
    assert.equal(typeof layer.weights.stats.sparsity, 'number');
  });

  test('values included by default', async () => {
    const { layers } = await peekLayers(FIXTURE);
    for (const layer of layers) {
      assert.ok(Array.isArray(layer.weights.values));
      assert.ok(Array.isArray(layer.bias.values));
    }
  });

  test('values: false returns null', async () => {
    const { layers } = await peekLayers(FIXTURE, { values: false });
    for (const layer of layers) {
      assert.equal(layer.weights.values, null);
      assert.equal(layer.bias.values, null);
    }
  });
});
