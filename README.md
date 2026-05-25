# tf-peek

Inspect TensorFlow.js model architecture and weights. Surfaces layer configs, weight matrices, biases, and computed stats (min, max, mean, std, sparsity) as structured JSON.

## Debug your model with a coding agent

Run `tf-peek` against your model and share the output with your assistant to get architectural analysis, flag dead neurons, spot regularisation gaps, and more.

```bash
npx tf-peek -modelPath=file://./path/to/model.json
```

**Claude Code** — add the skill so Claude can run it automatically:

```
Copy .claude/skills/tf-peek/ into your project's .claude/skills/ directory, then ask:
"Help me understand and flag potential issues in my model architecture"
```

Things an assistant can surface from the output:

- Layer shapes, activation functions, and regularisation config
- Weight magnitude and distribution per layer (`std`, `min`, `max`)
- Dead or near-zero weight rows (`sparsity`)
- Regularisation gaps (e.g. output layer missing L2)
- Bottleneck pressure (output layer compensating for weak hidden representations)

---

## CLI

```bash
npx tf-peek -modelPath=file://./path/to/model.json
```

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

## Module

```bash
npm install tf-peek
```

```typescript
import { peekLayers, ModelData } from 'tf-peek';

const data: ModelData = await peekLayers('file://./path/to/model.json');

console.log(data.model.totalParams);

for (const { name, config, weights, bias } of data.layers) {
  console.log(name, config.activation, weights.stats, bias.stats);
}
```
