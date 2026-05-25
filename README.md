# tf-peek [![Test](https://github.com/kunal-mandalia/tf-peek/actions/workflows/test.yml/badge.svg)](https://github.com/kunal-mandalia/tf-peek/actions/workflows/test.yml)

![tf-peek reveals hidden layers](logo.png)

TensorFlow.js model weights are opaque by default — you can train, save, and load them without ever seeing what they actually learned. For small models, inspecting the raw weights and biases can reveal skewed priors, dead neurons, and regularisation gaps that aggregate metrics like loss and accuracy miss entirely.

`tf-peek` surfaces layer configs, weight matrices, biases, and computed stats (min, max, mean, std, sparsity) as structured JSON or markdown.

## Discuss your DNN

Reason about your deep neural network (dnn) with a coding agent:

> *"I think there's something wrong with my output layer, take a look"*

> Here's the output layer (`dense_Dense3`) bias:
> ```
> [ 0.01, -0.21,  0.23 ]
>   Home   Draw   Away
> ```
> | Class    | Bias  | Effect     |
> |----------|-------|------------|
> | Home Win | +0.01 | Neutral    |
> | Draw     | −0.21 | Suppressed |
> | Away Win | +0.23 | Boosted    |
>
> The model has learned a skewed prior. Applying softmax to just the biases gives an implicit baseline of ~33% Home / ~26% Draw / ~41% Away — away wins are over-represented relative to real-world frequencies (~30%), and draws are under-represented. The `std` of 0.18 across only 3 values confirms the bias is doing real work, not sitting near zero. Worth investigating if the model is miscalibrated on draws.

---

## Getting started

Copy `.claude/skills/tf-peek/` into your project's `.claude/skills/` directory, then ask Claude Code:

> "Help me understand and flag potential issues in my model architecture"

Claude will run `tf-peek` against your model and reason about the architecture, weights and biases.

---

## CLI

```bash
npx tf-peek -modelPath=./path/to/model.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `-modelPath` | — | Path to `model.json` (required) |
| `-values` | `true` | Set to `false` to omit weight and bias arrays. Stats are always included. |
| `-format` | `json` | Output format: `json` or `markdown` |

Output:

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
      "config": { "units": 6, "activation": "relu", ... },
      "weights": {
        "shape": [8, 6],
        "values": [[...]],
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

To focus on metrics only (useful for large models):

```bash
npx tf-peek -modelPath=./path/to/model.json -values=false
```

`values` will be `null` in the output when omitted — `stats` are always computed.

## Module

```bash
npm install tf-peek
```

```typescript
import { peekLayers, ModelData } from 'tf-peek';

const data: ModelData = await peekLayers('./path/to/model.json');
// or, stats only:
const data: ModelData = await peekLayers('./path/to/model.json', { values: false });

console.log(data.model.totalParams);

for (const { name, config, weights, bias } of data.layers) {
  console.log(name, config.activation, weights.stats, bias.stats);
  // weights.values and bias.values are null when { values: false }
}
```

## Example output
```bash
% npx tf-peek --modelPath=example/model-3pl/model.json --format=markdown
```

### football-predict

|                       |           |
|-----------------------|-----------|
| Total params          | 97        |
| Input shape           | [null, 8] |
| Output shape          | [null, 3] |
| Trainable weights     | 6         |
| Non-trainable weights | 0         |

#### dense_Dense1

units: 6, activation: relu, useBias: true, name: dense_Dense1, trainable: true, dtype: float32

|         | shape  | min   | max  | mean | std  | sparsity |
|---------|--------|-------|------|------|------|----------|
| weights | [8, 6] | -0.48 | 0.58 | 0.06 | 0.22 | 0.04     |
| bias    | [6]    | 0.05  | 0.17 | 0.12 | 0.04 | 0        |

**weights**
```
[-0.11,  0.54, -0.01, -0.23,  0.16, -0.05]
[ 0.31, -0.01,  0.06,  0.45,  0.01, -0.48]
[-0.23,  0.35, -0.07, -0.01,  0.14,  0.12]
[ 0.07,  0.55, -0.12, -0.27,  0.18,  0.58]
[ 0.33,  0.08,  0.05, -0.12,     0, -0.24]
[-0.29,  0.05, -0.07, -0.29,  0.13,  0.46]
[ 0.05,  0.03,  0.03,  0.17,  0.01,  0.15]
[ 0.08,  0.04,  0.03,  0.11, -0.09,  0.17]
```

**bias**
```
[0.14, 0.09, 0.05, 0.12, 0.14, 0.17]
```

#### dense_Dense2

units: 4, activation: relu, useBias: true, name: dense_Dense2, trainable: true

|         | shape  | min   | max  | mean | std  | sparsity |
|---------|--------|-------|------|------|------|----------|
| weights | [6, 4] | -0.29 | 0.81 | 0.09 | 0.28 | 0.08     |
| bias    | [4]    | -0.04 | 0.26 | 0.1  | 0.12 | 0.25     |

**weights**
```
[ 0.14,  0.43, -0.15,  0.02]
[ 0.56, -0.29,   0.1,  0.81]
[ 0.06,  0.12,     0,  0.01]
[-0.27,  0.42,  -0.2, -0.06]
[ 0.04, -0.12,  0.02,  0.28]
[-0.12,  -0.2,  0.58, -0.01]
```

**bias**
```
[-0.04,  0.26,  0.17,  0.01]
```

#### dense_Dense3

units: 3, activation: softmax, useBias: true, name: dense_Dense3, trainable: true

|         | shape  | min   | max  | mean | std  | sparsity |
|---------|--------|-------|------|------|------|----------|
| weights | [4, 3] | -0.99 | 0.95 | 0.05 | 0.67 | 0        |
| bias    | [3]    | -0.21 | 0.23 | 0.01 | 0.18 | 0        |

**weights**
```
[ 0.13,  0.77, -0.67]
[-0.72, -0.24,  0.94]
[ 0.64, -0.99, -0.44]
[ 0.95,  0.59, -0.31]
```

**bias**
```
[ 0.01, -0.21,  0.23]
```