/**
 * Node.js Test Suite for Prediction Mode ONNX Models
 *
 * Tests inference with PredictionCausalLM exported models that accept:
 * - input_ids: [batch, seq_len]
 * - prediction_mask: [batch, seq_len, seq_len]
 * - evaluated_ids: [batch, seq_len]
 *
 * Returns:
 * - logits: [batch, num_evaluated, vocab_size]
 */

const ort = require('onnxruntime-node');
const path = require('path');
const fs = require('fs');

// Configuration
const CONFIG = {
	// Update this path after exporting a prediction mode model
	modelPath: path.resolve(__dirname, '../../outputs/trigor/20251115-trigo-gpt2-l6-d64-251112-invsqrt/GPT2CausalLM_ep0015_tree.onnx'),
	vocabSize: 259,
	// NOTE: These must match the exported model's fixed dimensions
	// Model was exported with: --prefix-len 10 --seq-len 15 (so eval_len=5)
	fixedPrefixLen: 10,
	fixedEvalLen: 5,
	tests: {
		basicInference: false,  // Disabled, dimensions don't match
		variableEvaluated: false,  // Disabled
		diagonalMask: false,  // Disabled, needs updated signature
		treeAttention: true,  // Tree attention test
	}
};

/**
 * Create bidirectional prediction mask
 * @param {number} seqLen - Total sequence length
 * @param {number} prefixLen - Length of prefix (causal region)
 * @returns {Float32Array} - Flattened mask [seqLen, seqLen]
 */
function createBidirectionalMask(seqLen, prefixLen) {
	const mask = new Float32Array(seqLen * seqLen);

	for (let i = 0; i < seqLen; i++) {
		for (let j = 0; j < seqLen; j++) {
			const idx = i * seqLen + j;

			if (i < prefixLen) {
				// Prefix region: causal mask
				mask[idx] = (j <= i) ? 1.0 : 0.0;
			} else {
				// Prediction region: bidirectional (all 1s)
				mask[idx] = 1.0;
			}
		}
	}

	return mask;
}

/**
 * Create diagonal prediction mask (each prediction token only sees itself + prefix)
 * @param {number} seqLen - Total sequence length
 * @param {number} prefixLen - Length of prefix (causal region)
 * @returns {Float32Array} - Flattened mask [seqLen, seqLen]
 */
function createDiagonalMask(seqLen, prefixLen) {
	const mask = new Float32Array(seqLen * seqLen);

	for (let i = 0; i < seqLen; i++) {
		for (let j = 0; j < seqLen; j++) {
			const idx = i * seqLen + j;

			if (i < prefixLen) {
				// Prefix region: causal mask
				mask[idx] = (j <= i) ? 1.0 : 0.0;
			} else {
				// Prediction region: diagonal (only self + all prefix)
				mask[idx] = (j < prefixLen || i === j) ? 1.0 : 0.0;
			}
		}
	}

	return mask;
}

/**
 * Create random input IDs
 * @param {number} batchSize
 * @param {number} seqLen
 * @returns {BigInt64Array}
 */
function createRandomInputIds(batchSize, seqLen) {
	const inputIds = new BigInt64Array(batchSize * seqLen);
	for (let i = 0; i < inputIds.length; i++) {
		inputIds[i] = BigInt(Math.floor(Math.random() * CONFIG.vocabSize));
	}
	return inputIds;
}

/**
 * Create evaluated_ids mask (marks which positions to return)
 * @param {number} batchSize
 * @param {number} seqLen
 * @param {number} prefixLen
 * @param {Array<number>} evaluatedPositions - If null, evaluate all prediction tokens
 * @returns {BigInt64Array}
 */
function createEvaluatedIds(batchSize, seqLen, prefixLen, evaluatedPositions = null) {
	const evaluatedIds = new BigInt64Array(batchSize * seqLen);

	for (let b = 0; b < batchSize; b++) {
		for (let i = 0; i < seqLen; i++) {
			const idx = b * seqLen + i;

			if (evaluatedPositions) {
				// Evaluate only specified positions
				evaluatedIds[idx] = evaluatedPositions.includes(i) ? 1n : 0n;
			} else {
				// Evaluate all prediction tokens
				evaluatedIds[idx] = (i >= prefixLen) ? 1n : 0n;
			}
		}
	}

	return evaluatedIds;
}

/**
 * Test basic inference with prediction mode
 */
async function testBasicInference(session) {
	console.log('\n' + '='.repeat(80));
	console.log('TEST 1: Basic Inference (Prediction Mode)');
	console.log('='.repeat(80));

	const batchSize = 1;
	const seqLen = CONFIG.fixedSeqLen;
	const prefixLen = CONFIG.fixedPrefixLen;

	console.log(`Running: batch=${batchSize}, seq_len=${seqLen}, prefix_len=${prefixLen}`);

	// Create inputs
	const inputIds = createRandomInputIds(batchSize, seqLen);
	const predictionMask = createBidirectionalMask(seqLen, prefixLen);
	const evaluatedIds = createEvaluatedIds(batchSize, seqLen, prefixLen);

	// Create ONNX tensors
	const inputIdsTensor = new ort.Tensor('int64', inputIds, [batchSize, seqLen]);
	const predictionMaskTensor = new ort.Tensor('float32', predictionMask, [1, seqLen, seqLen]);
	const evaluatedIdsTensor = new ort.Tensor('int64', evaluatedIds, [batchSize, seqLen]);

	// Run inference
	const startTime = Date.now();
	const results = await session.run({
		input_ids: inputIdsTensor,
		prediction_mask: predictionMaskTensor,
		evaluated_ids: evaluatedIdsTensor
	});
	const inferenceTime = Date.now() - startTime;

	const logits = results.logits;

	const numEvaluated = seqLen - prefixLen;
	const expectedShape = [batchSize, numEvaluated, CONFIG.vocabSize];

	console.log(`  Input shapes:`);
	console.log(`    input_ids: [${inputIdsTensor.dims.join(', ')}]`);
	console.log(`    prediction_mask: [${predictionMaskTensor.dims.join(', ')}]`);
	console.log(`    evaluated_ids: [${evaluatedIdsTensor.dims.join(', ')}]`);
	console.log(`  Output shape: [${logits.dims.join(', ')}]`);
	console.log(`  Expected shape: [${expectedShape.join(', ')}]`);
	console.log(`  Inference time: ${inferenceTime}ms`);

	// Validate shape
	const shapeMatch = logits.dims.length === 3 &&
	                   logits.dims[0] === batchSize &&
	                   logits.dims[1] === numEvaluated &&
	                   logits.dims[2] === CONFIG.vocabSize;

	if (shapeMatch) {
		console.log(`  ✓ Test passed`);

		// Show sample logits
		const sampleLogits = Array.from(logits.data.slice(0, 10));
		console.log(`  Sample logits (first 10): [${sampleLogits.map(v => v.toFixed(3)).join(', ')}]`);

		// Check logits range
		const logitsArray = Array.from(logits.data);
		const minLogit = Math.min(...logitsArray);
		const maxLogit = Math.max(...logitsArray);
		console.log(`  Logits range: [${minLogit.toFixed(3)}, ${maxLogit.toFixed(3)}]`);

		return true;
	} else {
		console.log(`  ✗ Test failed: Shape mismatch`);
		return false;
	}
}

/**
 * Test with variable evaluated positions
 */
async function testVariableEvaluated(session) {
	console.log('\n' + '='.repeat(80));
	console.log('TEST 2: Variable Evaluated Positions');
	console.log('='.repeat(80));

	const batchSize = 1;  // Match fixed model batch size
	const seqLen = CONFIG.fixedSeqLen;
	const prefixLen = CONFIG.fixedPrefixLen;

	// Evaluate only specific positions in prediction region
	const evaluatedPositions = [prefixLen + 3, prefixLen + 8, prefixLen + 13, prefixLen + 18];
	const numEvaluated = evaluatedPositions.length;

	console.log(`Running: batch=${batchSize}, seq_len=${seqLen}, prefix_len=${prefixLen}`);
	console.log(`Evaluating positions: ${evaluatedPositions.join(', ')}`);

	// Create inputs
	const inputIds = createRandomInputIds(batchSize, seqLen);
	const predictionMask = createBidirectionalMask(seqLen, prefixLen);
	const evaluatedIds = createEvaluatedIds(batchSize, seqLen, prefixLen, evaluatedPositions);

	// Create ONNX tensors (broadcast mask across batches)
	const inputIdsTensor = new ort.Tensor('int64', inputIds, [batchSize, seqLen]);
	const predictionMaskTensor = new ort.Tensor('float32', predictionMask, [1, seqLen, seqLen]);
	const evaluatedIdsTensor = new ort.Tensor('int64', evaluatedIds, [batchSize, seqLen]);

	// Run inference
	const startTime = Date.now();
	const results = await session.run({
		input_ids: inputIdsTensor,
		prediction_mask: predictionMaskTensor,
		evaluated_ids: evaluatedIdsTensor
	});
	const inferenceTime = Date.now() - startTime;

	const logits = results.logits;

	const expectedShape = [batchSize, numEvaluated, CONFIG.vocabSize];

	console.log(`  Output shape: [${logits.dims.join(', ')}]`);
	console.log(`  Expected shape: [${expectedShape.join(', ')}]`);
	console.log(`  Inference time: ${inferenceTime}ms`);

	// Validate shape
	const shapeMatch = logits.dims.length === 3 &&
	                   logits.dims[0] === batchSize &&
	                   logits.dims[1] === numEvaluated &&
	                   logits.dims[2] === CONFIG.vocabSize;

	if (shapeMatch) {
		console.log(`  ✓ Test passed`);
		return true;
	} else {
		console.log(`  ✗ Test failed: Shape mismatch`);
		return false;
	}
}

/**
 * Test with diagonal mask pattern
 */
async function testDiagonalMask(session) {
	console.log('\n' + '='.repeat(80));
	console.log('TEST 3: Diagonal Mask Pattern');
	console.log('='.repeat(80));

	const batchSize = 1;
	const prefixLen = CONFIG.fixedPrefixLen;
	const evalLen = CONFIG.fixedEvalLen;
	const seqLen = prefixLen + evalLen;

	console.log(`Running: batch=${batchSize}, seq_len=${seqLen}, prefix_len=${prefixLen}`);
	console.log(`Mask type: Diagonal (each prediction token only sees itself + prefix)`);

	// Create inputs with diagonal mask
	const inputIds = createRandomInputIds(batchSize, seqLen);
	const predictionMask = createDiagonalMask(seqLen, prefixLen);  // Diagonal instead of bidirectional
	const evaluatedIds = createEvaluatedIds(batchSize, seqLen, prefixLen);

	// Create ONNX tensors
	const inputIdsTensor = new ort.Tensor('int64', inputIds, [batchSize, seqLen]);
	const predictionMaskTensor = new ort.Tensor('float32', predictionMask, [1, seqLen, seqLen]);
	const evaluatedIdsTensor = new ort.Tensor('int64', evaluatedIds, [batchSize, seqLen]);

	// Run inference
	const startTime = Date.now();
	const results = await session.run({
		input_ids: inputIdsTensor,
		prediction_mask: predictionMaskTensor,
		evaluated_ids: evaluatedIdsTensor
	});
	const inferenceTime = Date.now() - startTime;

	const logits = results.logits;

	const numEvaluated = evalLen;
	const expectedShape = [batchSize, numEvaluated, CONFIG.vocabSize];

	console.log(`  Output shape: [${logits.dims.join(', ')}]`);
	console.log(`  Inference time: ${inferenceTime}ms`);

	const shapeMatch = logits.dims.length === 3 &&
	                   logits.dims[0] === batchSize &&
	                   logits.dims[1] === numEvaluated;

	if (shapeMatch) {
		console.log(`  ✓ Test passed`);
		return true;
	} else {
		console.log(`  ✗ Test failed`);
		return false;
	}
}

/**
 * Test tree attention pattern (multiple causal sequences branching from common root)
 */
async function testTreeAttention(session) {
	console.log('\n' + '='.repeat(80));
	console.log('TEST 4: Tree Attention Pattern');
	console.log('='.repeat(80));

	const batchSize = 1;
	const prefixLen = 10;  // Small prefix for context
	const m = 5;  // 5 evaluated tokens: [a, b, c, d, e]

	console.log(`Testing tree attention with 2 causal branches:`);
	console.log(`  Branch 1: a → b → c`);
	console.log(`  Branch 2: a → d → e`);
	console.log(`  Common root: a`);

	// Create prefix and evaluated tokens
	const prefixIds = createRandomInputIds(batchSize, prefixLen);
	const evaluatedIds = createRandomInputIds(batchSize, m);

	// Create tree attention mask for evaluated region
	// Positions: [a=0, b=1, c=2, d=3, e=4]
	// Tree structure:
	//        a (root)
	//       / \
	//      b   d
	//      |   |
	//      c   e
	const treeMask = new Float32Array(m * m);

	// Initialize all to 0
	for (let i = 0; i < m * m; i++) {
		treeMask[i] = 0.0;
	}

	// Position 0 (a): attends to itself
	treeMask[0 * m + 0] = 1.0;

	// Position 1 (b): attends to a, b (branch 1 path)
	treeMask[1 * m + 0] = 1.0;  // a
	treeMask[1 * m + 1] = 1.0;  // b

	// Position 2 (c): attends to a, b, c (branch 1 path)
	treeMask[2 * m + 0] = 1.0;  // a
	treeMask[2 * m + 1] = 1.0;  // b
	treeMask[2 * m + 2] = 1.0;  // c

	// Position 3 (d): attends to a, d (branch 2 path, NOT b or c)
	treeMask[3 * m + 0] = 1.0;  // a
	treeMask[3 * m + 3] = 1.0;  // d

	// Position 4 (e): attends to a, d, e (branch 2 path, NOT b or c)
	treeMask[4 * m + 0] = 1.0;  // a
	treeMask[4 * m + 3] = 1.0;  // d
	treeMask[4 * m + 4] = 1.0;  // e

	console.log(`\n  Attention mask structure (evaluated region ${m}×${m}):`);
	console.log(`       a  b  c  d  e`);
	for (let i = 0; i < m; i++) {
		const rowLabel = ['a', 'b', 'c', 'd', 'e'][i];
		let row = `    ${rowLabel} [`;
		for (let j = 0; j < m; j++) {
			row += ` ${treeMask[i * m + j].toFixed(0)} `;
		}
		row += ']';
		console.log(row);
	}

	// Create ONNX tensors
	const prefixIdsTensor = new ort.Tensor('int64', prefixIds, [batchSize, prefixLen]);
	const evaluatedIdsTensor = new ort.Tensor('int64', evaluatedIds, [batchSize, m]);
	const evaluatedMaskTensor = new ort.Tensor('float32', treeMask, [1, m, m]);

	// Run inference
	console.log(`\n  Running inference...`);
	const startTime = Date.now();
	const results = await session.run({
		prefix_ids: prefixIdsTensor,
		evaluated_ids: evaluatedIdsTensor,
		evaluated_mask: evaluatedMaskTensor
	});
	const inferenceTime = Date.now() - startTime;

	const logits = results.logits;

	console.log(`  Inference time: ${inferenceTime}ms`);
	console.log(`  Output shape: [${logits.dims.join(', ')}]`);
	console.log(`  Expected: [${batchSize}, ${m + 1}, ${CONFIG.vocabSize}]`);

	// Extract logits for each position (skip last prefix position at index 0)
	const vocabSize = CONFIG.vocabSize;

	// Helper to compute softmax for a single position
	function softmax(logitsArray, startIdx, vocabSize) {
		const probs = new Array(vocabSize);
		let maxLogit = -Infinity;

		// Find max for numerical stability
		for (let i = 0; i < vocabSize; i++) {
			maxLogit = Math.max(maxLogit, logitsArray[startIdx + i]);
		}

		// Compute exp and sum
		let sumExp = 0;
		for (let i = 0; i < vocabSize; i++) {
			probs[i] = Math.exp(logitsArray[startIdx + i] - maxLogit);
			sumExp += probs[i];
		}

		// Normalize
		for (let i = 0; i < vocabSize; i++) {
			probs[i] /= sumExp;
		}

		return probs;
	}

	// Get token IDs for evaluated positions
	const evalTokens = [];
	for (let i = 0; i < m; i++) {
		// evaluatedIds is BigInt64Array
		evalTokens.push(Number(evaluatedIds[i]));
	}

	console.log(`\n  Token IDs: [${evalTokens.join(', ')}]`);

	// Compute probabilities for each position
	const logitsArray = Array.from(logits.data);
	const probabilities = [];

	for (let i = 0; i < m; i++) {
		// Position i+1 in output (i+1 because index 0 is last prefix)
		const startIdx = (i + 1) * vocabSize;
		const probs = softmax(logitsArray, startIdx, vocabSize);
		const tokenId = evalTokens[i];
		const tokenProb = probs[tokenId];
		probabilities.push(tokenProb);
	}

	console.log(`\n  Token probabilities:`);
	for (let i = 0; i < m; i++) {
		const label = ['a', 'b', 'c', 'd', 'e'][i];
		console.log(`    P(${label}|context) = ${probabilities[i].toFixed(6)}`);
	}

	// Compute sequence probabilities
	// Sequence 1: a → b → c
	const seq1Prob = probabilities[0] * probabilities[1] * probabilities[2];

	// Sequence 2: a → d → e
	const seq2Prob = probabilities[0] * probabilities[3] * probabilities[4];

	console.log(`\n  Sequence probabilities:`);
	console.log(`    Branch 1 (a→b→c): P(a) × P(b|a) × P(c|a,b) = ${seq1Prob.toExponential(4)}`);
	console.log(`    Branch 2 (a→d→e): P(a) × P(d|a) × P(e|a,d) = ${seq2Prob.toExponential(4)}`);

	const probRatio = seq1Prob / seq2Prob;
	console.log(`    Ratio (branch1/branch2): ${probRatio.toFixed(4)}`);

	// Validation
	const shapeMatch = logits.dims[0] === batchSize &&
	                   logits.dims[1] === m + 1 &&
	                   logits.dims[2] === vocabSize;

	const probsValid = probabilities.every(p => p > 0 && p <= 1.0 && !isNaN(p));

	if (shapeMatch && probsValid) {
		console.log(`\n  ✓ Tree attention test passed`);
		return true;
	} else {
		console.log(`\n  ✗ Tree attention test failed`);
		if (!shapeMatch) console.log(`    - Shape mismatch`);
		if (!probsValid) console.log(`    - Invalid probabilities`);
		return false;
	}
}

/**
 * Run all tests
 */
async function runTests() {
	console.log('\n' + '='.repeat(80));
	console.log('ONNX Prediction Mode Inference Test Suite (Node.js)');
	console.log('='.repeat(80));

	// Check if model file exists
	if (!fs.existsSync(CONFIG.modelPath)) {
		console.error(`\n✗ Model file not found: ${CONFIG.modelPath}`);
		console.error('\nTo generate the model, run:');
		console.error('  python exportOnnx.py <training_dir> --prediction-mode --checkpoint best\n');
		process.exit(1);
	}

	console.log(`\nModel: ${path.basename(CONFIG.modelPath)}`);
	const stats = fs.statSync(CONFIG.modelPath);
	console.log(`Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);

	// Create inference session
	console.log('\nCreating inference session...');
	const session = await ort.InferenceSession.create(CONFIG.modelPath);
	console.log('✓ Session created');

	const results = {
		passed: 0,
		failed: 0
	};

	// Run tests
	if (CONFIG.tests.basicInference) {
		const passed = await testBasicInference(session);
		if (passed) results.passed++; else results.failed++;
	}

	if (CONFIG.tests.variableEvaluated) {
		const passed = await testVariableEvaluated(session);
		if (passed) results.passed++; else results.failed++;
	}

	if (CONFIG.tests.diagonalMask) {
		const passed1 = await testDiagonalMask(session);
		if (passed1) results.passed++; else results.failed++;
	}

	// Test tree attention
	if (CONFIG.tests.treeAttention) {
		const passed2 = await testTreeAttention(session);
		if (passed2) results.passed++; else results.failed++;
	}

	// Summary
	console.log('\n' + '='.repeat(80));
	console.log('Test Summary');
	console.log('='.repeat(80));
	console.log(`Passed: ${results.passed}`);
	console.log(`Failed: ${results.failed}`);
	console.log(`Total: ${results.passed + results.failed}`);
	console.log('='.repeat(80) + '\n');

	if (results.failed > 0) {
		process.exit(1);
	}
}

// Run tests
runTests().catch(err => {
	console.error('\n✗ Error running tests:');
	console.error(err);
	process.exit(1);
});
