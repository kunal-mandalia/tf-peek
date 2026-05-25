---
name: tf-peek
description: |
 Describes the layers of model, its weights and bias as compactly as possible. Use this to understand / get context on a model
---

# Skill: tf-peek

Use this skill to get context on a particular model

Inspects a TensorFlow.js model: loads it and prints all layer weights and biases as JSON to stdout.

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

JSON array of layers, each with weights and biases:

```json
[
  {
    "name": "dense_Dense1",
    "config": {
      "units": 3,
      "activation": "softmax",
      "useBias": true,
      ...
    },
    "weights": { "shape": [4, 8], "values": [[...], ...] },
    "bias": { "shape": [8], "values": [...] }
  }
]
```

---

## Module Usage

```typescript
import { peekLayers, LayerData } from 'tf-peek';

const layers: LayerData[] = await peekLayers('file://./models/classifier/model.json');

// layers is an array — iterate, filter, or pass to downstream logic
for (const { layer, weights, bias } of layers) {
  console.log(layer.name, weights.shape, bias.values);
}
```
