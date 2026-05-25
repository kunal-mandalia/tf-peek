import tf, { loadLayersModel } from "@tensorflow/tfjs"
import { readFileSync } from "fs"
import { dirname, resolve } from "path"

function fileIOHandler(modelPath: string): tf.io.IOHandler {
    const filePath = modelPath.replace(/^file:\/\//, '')
    const modelDir = dirname(filePath)
    return {
        load: async (): Promise<tf.io.ModelArtifacts> => {
            const modelJson = JSON.parse(readFileSync(filePath, 'utf8'))
            const weightBuffers: Buffer[] = modelJson.weightsManifest.flatMap(
                (group: { paths: string[] }) => group.paths.map(
                    (p: string) => readFileSync(resolve(modelDir, p))
                )
            )
            const weightData = Buffer.concat(weightBuffers)
            return {
                modelTopology: modelJson.modelTopology,
                weightSpecs: modelJson.weightsManifest.flatMap(
                    (g: { weights: tf.io.WeightsManifestEntry[] }) => g.weights
                ),
                weightData: weightData.buffer.slice(
                    weightData.byteOffset,
                    weightData.byteOffset + weightData.byteLength
                ),
                format: modelJson.format,
                generatedBy: modelJson.generatedBy,
                convertedBy: modelJson.convertedBy,
            }
        }
    }
}

async function loadModel(modelPath: string) {
    tf.env().set('IS_NODE', false)
    await tf.ready()
    return loadLayersModel(fileIOHandler(modelPath))
}

function precision1D(row: number[], dp: number) {
    return row.map(r => parseFloat(r.toFixed(dp)))
}

function precision2D(matrix: number[][], dp: number) {
    let output: number[][] = []
    for (let i=0; i<matrix.length; i++) {
        const row = matrix[i]
        const precRow: number[] = row.map(r => parseFloat(r.toFixed(dp)))
        output.push(precRow)
    }
    return output
}

export type WeightStats = {
    min: number,
    max: number,
    mean: number,
    std: number,
    sparsity: number,
}

function computeStats(values: number[], dp: number): WeightStats {
    const n = values.length
    const min = Math.min(...values)
    const max = Math.max(...values)
    const mean = values.reduce((s, v) => s + v, 0) / n
    const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / n)
    const sparsity = values.filter(v => Math.abs(v) < 0.01).length / n
    return {
        min: parseFloat(min.toFixed(dp)),
        max: parseFloat(max.toFixed(dp)),
        mean: parseFloat(mean.toFixed(dp)),
        std: parseFloat(std.toFixed(dp)),
        sparsity: parseFloat(sparsity.toFixed(dp)),
    }
}

export type ModelInfo = {
    name: string,
    totalParams: number,
    inputShape: (number | null)[],
    outputShape: (number | null)[],
    trainableWeights: number,
    nonTrainableWeights: number,
}

export type PeekOptions = {
    values?: boolean,
    format?: 'json' | 'markdown',
}

export type LayerData = {
    name: string,
    config: tf.serialization.ConfigDict,
    weights: { shape: number[], values: number[][] | null, stats: WeightStats },
    bias: { shape: number[], values: number[] | null, stats: WeightStats },
}

export type ModelData = {
    model: ModelInfo,
    layers: LayerData[],
}

function extractModel(model: tf.LayersModel, options: PeekOptions = {}): ModelData {
    const includeValues = options.values !== false
    const layers: LayerData[] = []

    model.layers.forEach((layer) => {
        const config = layer.getConfig()

        if (layer.weights.length === 0) return

        const weightsTensor = layer.weights[0].read()
        const biasesTensor = layer.weights[1].read()
        const weightsValues = weightsTensor.arraySync() as number[][]
        const biasValues = biasesTensor.arraySync() as number[]

        layers.push({
            name: layer.name,
            config,
            weights: {
                shape: weightsTensor.shape,
                values: includeValues ? precision2D(weightsValues, 2) : null,
                stats: computeStats(weightsValues.flat(), 2),
            },
            bias: {
                shape: biasesTensor.shape,
                values: includeValues ? precision1D(biasValues, 2) : null,
                stats: computeStats(biasValues, 2),
            },
        })
    })

    return {
        model: {
            name: model.name,
            totalParams: model.countParams(),
            inputShape: model.inputs[0].shape,
            outputShape: model.outputs[0].shape,
            trainableWeights: model.trainableWeights.length,
            nonTrainableWeights: model.nonTrainableWeights.length,
        },
        layers,
    }
}

export async function peekLayers(modelPath: string, options: PeekOptions = {}): Promise<ModelData> {
    const model = await loadModel(modelPath)
    return extractModel(model, options)
}

function markdownTable(rows: string[][]): string {
    const widths = rows[0].map((_, c) => Math.max(...rows.map(r => r[c].length)))
    const fmt = (row: string[]) => '| ' + row.map((cell, c) => cell.padEnd(widths[c])).join(' | ') + ' |'
    const sep = '|' + widths.map(w => '-'.repeat(w + 2)).join('|') + '|'
    const [header, ...body] = rows
    return [fmt(header), sep, ...body.map(fmt)].join('\n')
}

function shape(s: (number | null)[]): string {
    return '[' + s.map(v => v ?? 'null').join(', ') + ']'
}

function formatMarkdown(data: ModelData): string {
    const { model, layers } = data
    const lines: string[] = []

    lines.push(`# ${model.name}`, '')
    lines.push(markdownTable([
        ['', ''],
        ['Total params', String(model.totalParams)],
        ['Input shape', shape(model.inputShape)],
        ['Output shape', shape(model.outputShape)],
        ['Trainable weights', String(model.trainableWeights)],
        ['Non-trainable weights', String(model.nonTrainableWeights)],
    ]), '')

    for (const { name, config, weights, bias } of layers) {
        lines.push(`## ${name}`, '')

        const configStr = Object.entries(config)
            .filter(([, v]) => v !== null && v !== undefined && typeof v !== 'object')
            .map(([k, v]) => `${k}: ${v}`)
            .join(', ')
        if (configStr) lines.push(configStr, '')

        const s = (x: WeightStats) => [x.min, x.max, x.mean, x.std, x.sparsity].map(String)
        lines.push(markdownTable([
            ['', 'shape', 'min', 'max', 'mean', 'std', 'sparsity'],
            ['weights', shape(weights.shape), ...s(weights.stats)],
            ['bias',    shape(bias.shape),    ...s(bias.stats)],
        ]))

        if (weights.values !== null) {
            const strs = weights.values.map(row => row.map(String))
            const colW = strs[0].map((_, c) => Math.max(...strs.map(r => r[c].length)))
            const rows = strs.map(row => '[' + row.map((v, c) => v.padStart(colW[c])).join(', ') + ']')
            lines.push('', '**weights**', '```', ...rows, '```')
        }
        if (bias.values !== null) {
            const strs = bias.values.map(String)
            const w = Math.max(...strs.map(s => s.length))
            lines.push('', '**bias**', '```', '[' + strs.map(s => s.padStart(w)).join(', ') + ']', '```')
        }
        lines.push('')
    }

    return lines.join('\n')
}

export async function toStdOut(modelPath: string, options: PeekOptions = {}) {
    const data = await peekLayers(modelPath, options)
    if (options.format === 'markdown') {
        console.log(formatMarkdown(data))
    } else {
        console.log(JSON.stringify(data, null, 2))
    }
}
