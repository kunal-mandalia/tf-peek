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

Path to the `model.json` file.

```
-modelPath=./relative/path/to/model.json
-modelPath=/absolute/path/to/model.json
```

### `-values`

Default: `true`. Set to `false` to omit weight and bias arrays — stats are always computed. Use this for large models where the full weight matrices would overwhelm the context.

### `-format`

Default: `json`. Set to `markdown` for human-readable output or when sharing with a coding agent — it is much more token-efficient than JSON for large models.

### Example

```bash
# Full JSON output
npx tf-peek -modelPath=./models/classifier/model.json

# Markdown summary, stats only (best for large models shared with an agent)
npx tf-peek -modelPath=./models/classifier/model.json -format=markdown -values=false
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
