import tf, { loadLayersModel } from "@tensorflow/tfjs-node"

async function loadModel(modelPath: string) {
    await tf.ready()
    return loadLayersModel(modelPath)
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

export type LayerData = {
    name: string,
    config: tf.serialization.ConfigDict,
    weights: { shape: number[], values: number[][] },
    bias: { shape: number[], values: number[] },
}

function extractLayers(model: tf.LayersModel): LayerData[] {
    const layers: LayerData[] = []

    model.layers.forEach((layer) => {
        const config = layer.getConfig()

        if (layer.weights.length === 0) return

        const weightsTensor = layer.weights[0].read()
        const biasesTensor = layer.weights[1].read()

        layers.push({
            name: layer.name,
            config,
            weights: {
                shape: weightsTensor.shape,
                values: precision2D(weightsTensor.arraySync() as number[][], 2),
            },
            bias: {
                shape: biasesTensor.shape,
                values: precision1D(biasesTensor.arraySync() as number[], 2),
            },
        })
    })
    return layers
}

function printLayers(layers: LayerData[]) {
    console.log(JSON.stringify(layers, null, 2))
}

export async function peekLayers(modelPath: string): Promise<LayerData[]> {
    const model = await loadModel(modelPath)
    return extractLayers(model)
}

export async function toStdOut(modelPath: string) {
    const layers = await peekLayers(modelPath)
    printLayers(layers)
}
