# Skill: tf-peek

Dual-purpose utility enabling programmatic string processing or raw CLI automation execution.

## Integration Protocols

### Protocol 1: Subprocess / Shell Tool (CLI Mode)

Invoke via shell execution layers. Output is strictly isolated to standard output stream (`stdout`).

```bash
npx tf-peek "input_string"
```

**Expected stdout output:**

```
Processed: input_string
```

### Protocol 2: Native Runtime Import (Module Mode)

Import core modules inside sandbox runtimes or JavaScript execution environments.

```javascript
import { coreUtility } from 'tf-peek';

const data = coreUtility('input_string');
```

**CommonJS:**

```javascript
const { coreUtility } = require('tf-peek');

const data = coreUtility('input_string');
```
