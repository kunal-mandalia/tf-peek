---
name: tf-peek
description: |
 Use this to understand or get context on a model.
 
 Inspects a TensorFlow.js model and outputs model-level info (name, param count, shapes), per-layer configs (activation, units, regularization), weights and biases with computed stats (min, max, mean, std, sparsity).
---

# Skill: tf-peek

Use this skill to get context on a particular model

Inspects a TensorFlow.js model: loads it and prints model info, layer configs, weights, biases, and computed stats as JSON to stdout.

## CLI Usage

```bash
npx tf-peek -modelPath=<path>
```

### `-modelPath`

Path to the `model.json` file. Must be prefixed with `file://` for local paths.

```
-modelPath=file:///absolute/path/to/model.json
-modelPath=file://./relative/path/to/model.json
```

### Example

```bash
npx tf-peek -modelPath=file://./models/classifier/model.json
```

### stdout

JSON object with top-level `model` info and a `layers` array:

```json
{
  "model": {
    "name": "my-model",
    "totalParams": 97,
    "inputShape": [null, 8],
    "outputShape": [null, 3],
    "trainableWeights": 6,
    "nonTrainableWeights": 0
  },
  "layers": [
    {
      "name": "dense_Dense1",
      "config": {
        "units": 6,
        "activation": "relu",
        "useBias": true,
        ...
      },
      "weights": {
        "shape": [8, 6],
        "values": [[...], ...],
        "stats": { "min": -0.48, "max": 0.58, "mean": 0.06, "std": 0.22, "sparsity": 0.04 }
      },
      "bias": {
        "shape": [6],
        "values": [...],
        "stats": { "min": 0.05, "max": 0.17, "mean": 0.12, "std": 0.04, "sparsity": 0 }
      }
    }
  ]
}
```

`stats.sparsity` is the fraction of weights with absolute value < 0.01.

---

## Module Usage

```typescript
import { peekLayers, ModelData } from 'tf-peek';

const data: ModelData = await peekLayers('file://./models/classifier/model.json');

console.log(data.model.totalParams);

for (const { name, config, weights, bias } of data.layers) {
  console.log(name, config.activation, weights.shape, weights.stats, bias.values);
}
```
