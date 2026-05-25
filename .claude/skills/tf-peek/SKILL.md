---
name: tf-peek
description: |
 Use this to understand or get context on a model.
 
 Inspects a TensorFlow.js model and outputs model-level info (name, param count, shapes), per-layer configs (activation, units, regularization), weights and biases with computed stats (min, max, mean, std, sparsity).
---

# Skill: tf-peek

Use this skill to get context on a particular model

Inspects a TensorFlow.js model: loads it and prints model info, layer configs, weights, biases, and computed stats to stdout.

## CLI Usage

```bash
npx tf-peek -modelPath=<path>
```

### `-modelPath`

Path to the `model.json` file.

```
-modelPath=./relative/path/to/model.json
-modelPath=/absolute/path/to/model.json
```

### `-values`

Default: `true`. Set to `false` to omit weight and bias arrays — stats are always computed.

### `-format`

Default: `markdown`.

- Use `markdown` to explore and reason about the model — aligned tables and matrices let you scan structure at a glance.
- Use `json` when you need to extract or filter specific data (e.g. read a single layer's weights, pipe to a script).

### Examples

#### stdout: markdown

```bash
npx tf-peek -modelPath=./models/classifier/model.json -format=markdown
```

```markdown

# football-predict

|                       |           |
|-----------------------|-----------|
| Total params          | 97        |
| Input shape           | [null, 8] |
| Output shape          | [null, 3] |
| Trainable weights     | 6         |
| Non-trainable weights | 0         |

## dense_Dense1

units: 6, activation: relu, useBias: true, name: dense_Dense1, trainable: true, dtype: float32

|         | shape  | min   | max  | mean | std  | sparsity |
|---------|--------|-------|------|------|------|----------|
| weights | [8, 6] | -0.48 | 0.58 | 0.06 | 0.22 | 0.04     |
| bias    | [6]    | 0.05  | 0.17 | 0.12 | 0.04 | 0        |

**weights**
[-0.11,  0.54, -0.01, -0.23,  0.16, -0.05]
[ 0.31, -0.01,  0.06,  0.45,  0.01, -0.48]
[-0.23,  0.35, -0.07, -0.01,  0.14,  0.12]
[ 0.07,  0.55, -0.12, -0.27,  0.18,  0.58]
[ 0.33,  0.08,  0.05, -0.12,     0, -0.24]
[-0.29,  0.05, -0.07, -0.29,  0.13,  0.46]
[ 0.05,  0.03,  0.03,  0.17,  0.01,  0.15]
[ 0.08,  0.04,  0.03,  0.11, -0.09,  0.17]

**bias**
[0.14, 0.09, 0.05, 0.12, 0.14, 0.17]

```

#### stdout: JSON

JSON object with top-level `model` info and a `layers` array:

```bash
npx tf-peek -modelPath=./models/classifier/model.json -format=json
```

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

## Troubleshooting

### `ENOVERSIONS` / No versions available

If `~/.npmrc` contains `min-release-age=<days>`, npm will refuse to install packages published fewer than that many days ago. `tf-peek` itself may trigger this if it was recently published.

Override for the command:

```bash
npm_config_min_release_age=0 npx tf-peek --modelPath=./models/classifier/model.json
```

Or add a shell alias so it always applies:

```bash
# ~/.zshrc
alias tf-peek='npm_config_min_release_age=0 npx tf-peek'
```

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
