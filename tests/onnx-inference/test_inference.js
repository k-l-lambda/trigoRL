/**
 * Node.js ONNX Model Inference Test
 *
 * This test uses the onnxruntime-node binding from third_party/onnxruntime
 * to test the exported GPT-2 ONNX model.
 *
 * Setup:
 *   cd tests/onnx-inference
 *   npm install
 *
 * Run:
 *   npm test
 */

const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');


/**
 * Test configuration
 */
const CONFIG = {
	modelPath: path.resolve(__dirname, '../../outputs/trigor/20251115-trigo-gpt2-l6-d64-251112-invsqrt/GPT2CausalLM_ep0015_int8.onnx'),
	vocabSize: 259,
	tests: {
		basicInference: true,
		batchSizes: [1],  // Fixed batch size
		seqLengths: [256],  // Fixed sequence length (ONNX export limitation with GPT-2 position embeddings)
		generation: true,  // Disabled: requires variable seq length
		generationTokens: 10,
	}
};


/**
 * Main test runner
 */
async function runTests() {
	console.log('='.repeat(80));
	console.log('ONNX Model Inference Test Suite (Node.js)');
	console.log('='.repeat(80));

	// Verify model exists
	if (!fs.existsSync(CONFIG.modelPath)) {
		throw new Error(`Model file not found: ${CONFIG.modelPath}`);
	}

	const stats = fs.statSync(CONFIG.modelPath);
	console.log(`\nModel: ${path.basename(CONFIG.modelPath)}`);
	console.log(`Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
	console.log(`Path: ${CONFIG.modelPath}`);

	// Create inference session
	console.log('\n' + '-'.repeat(80));
	console.log('Creating inference session...');
	const session = await ort.InferenceSession.create(CONFIG.modelPath, {
		executionProviders: ['cpu'],
		graphOptimizationLevel: 'all',
	});
	console.log('✓ Session created');

	// Display model info
	printModelInfo(session);

	// Run tests
	let passedTests = 0;
	let totalTests = 0;

	if (CONFIG.tests.basicInference) {
		console.log('\n' + '='.repeat(80));
		console.log('TEST 1: Basic Inference');
		console.log('='.repeat(80));
		totalTests++;
		try {
			await testBasicInference(session);
			passedTests++;
		} catch (error) {
			console.error('✗ Test failed:', error.message);
		}
	}

	if (CONFIG.tests.batchSizes.length > 0) {
		console.log('\n' + '='.repeat(80));
		console.log('TEST 2: Variable Batch Sizes');
		console.log('='.repeat(80));
		for (const batchSize of CONFIG.tests.batchSizes) {
			totalTests++;
			try {
				await testInference(session, batchSize, 256);
				passedTests++;
			} catch (error) {
				console.error(`✗ Test failed (batch=${batchSize}):`, error.message);
			}
		}
	}

	if (CONFIG.tests.seqLengths.length > 0) {
		console.log('\n' + '='.repeat(80));
		console.log('TEST 3: Variable Sequence Lengths');
		console.log('='.repeat(80));
		for (const seqLen of CONFIG.tests.seqLengths) {
			totalTests++;
			try {
				await testInference(session, 1, seqLen);
				passedTests++;
			} catch (error) {
				console.error(`✗ Test failed (seqLen=${seqLen}):`, error.message);
			}
		}
	}

	if (CONFIG.tests.generation) {
		console.log('\n' + '='.repeat(80));
		console.log('TEST 4: Autoregressive Generation');
		console.log('='.repeat(80));
		totalTests++;
		try {
			await testGeneration(session, CONFIG.tests.generationTokens);
			passedTests++;
		} catch (error) {
			console.error('✗ Test failed:', error.message);
		}
	}

	// Summary
	console.log('\n' + '='.repeat(80));
	console.log('Test Summary');
	console.log('='.repeat(80));
	console.log(`Total tests: ${totalTests}`);
	console.log(`Passed: ${passedTests}`);
	console.log(`Failed: ${totalTests - passedTests}`);

	if (passedTests === totalTests) {
		console.log('\n✓ All tests passed!');
		console.log('='.repeat(80));
		return true;
	} else {
		console.log('\n✗ Some tests failed!');
		console.log('='.repeat(80));
		return false;
	}
}


/**
 * Print model information
 */
function printModelInfo(session) {
	console.log('\n' + '-'.repeat(80));
	console.log('Model Information');
	console.log('-'.repeat(80));

	console.log('\nInputs:');
	session.inputNames.forEach((name, i) => {
		console.log(`  [${i}] ${name}`);
	});

	console.log('\nOutputs:');
	session.outputNames.forEach((name, i) => {
		console.log(`  [${i}] ${name}`);
	});
}


/**
 * Test basic inference
 */
async function testBasicInference(session) {
	const batchSize = 1;
	const seqLen = 256;

	console.log(`\nRunning: batch=${batchSize}, seq_len=${seqLen}`);

	// Create input tensor
	const inputIds = createRandomInput(batchSize, seqLen);
	const inputTensor = new ort.Tensor('int64', inputIds, [batchSize, seqLen]);

	// Run inference
	const startTime = Date.now();
	const results = await session.run({ input_ids: inputTensor });
	const duration = Date.now() - startTime;

	// Validate output
	const logits = results.logits;
	validateOutput(logits, batchSize, seqLen);

	// Print results
	console.log(`  Input shape: [${inputTensor.dims.join(', ')}]`);
	console.log(`  Output shape: [${logits.dims.join(', ')}]`);
	console.log(`  Output dtype: ${logits.type}`);
	console.log(`  Inference time: ${duration}ms`);

	// Get predictions
	const predictions = getPredictions(logits.data, batchSize * seqLen, CONFIG.vocabSize);
	console.log(`  Sample predictions: [${predictions.slice(0, 10).join(', ')}]`);

	const logitsArray = Array.from(logits.data);
	console.log(`  Logits range: [${Math.min(...logitsArray).toFixed(3)}, ${Math.max(...logitsArray).toFixed(3)}]`);

	console.log('  ✓ Test passed');
}


/**
 * Test inference with specific batch size and sequence length
 */
async function testInference(session, batchSize, seqLen) {
	console.log(`\nTesting: batch=${batchSize}, seq_len=${seqLen}`);

	const inputIds = createRandomInput(batchSize, seqLen);
	const inputTensor = new ort.Tensor('int64', inputIds, [batchSize, seqLen]);

	const startTime = Date.now();
	const results = await session.run({ input_ids: inputTensor });
	const duration = Date.now() - startTime;

	const logits = results.logits;
	validateOutput(logits, batchSize, seqLen);

	console.log(`  Output: [${logits.dims.join(', ')}], Time: ${duration}ms ✓`);
}


/**
 * Test autoregressive generation
 */
async function testGeneration(session, numTokens) {
	console.log(`\nGenerating ${numTokens} tokens autoregressively...`);

	// TGN tokenizer: byte-level (0-255) + PAD(256) + START(257) + END(258)
	const PAD_TOKEN = 256;
	const PROMPT = "[Board 5x5]";

	// Convert prompt to token IDs (byte values)
	const promptTokens = Array.from(PROMPT).map(c => c.charCodeAt(0));
	console.log(`  Prompt: "${PROMPT}"`);
	console.log(`  Prompt tokens (${promptTokens.length}): [${promptTokens.join(', ')}]`);

	// Start with prompt tokens
	const sequence = [...promptTokens];
	const times = [];

	// Generate tokens
	for (let i = 0; i < numTokens; i++) {
		// Pad sequence to fixed length 256
		const paddedLength = 256;
		const paddedSequence = [...sequence];
		while (paddedSequence.length < paddedLength) {
			paddedSequence.push(PAD_TOKEN);
		}

		// Create input tensor
		const inputIds = new BigInt64Array(paddedSequence.map(t => BigInt(t)));
		const inputTensor = new ort.Tensor('int64', inputIds, [1, paddedLength]);

		const startTime = Date.now();
		const results = await session.run({ input_ids: inputTensor });
		times.push(Date.now() - startTime);

		// Get prediction at the last non-padded position
		const logits = results.logits.data;
		const lastPos = sequence.length - 1;  // Position before padding
		const offset = lastPos * CONFIG.vocabSize;

		// Find token with highest logit
		let maxIdx = 0;
		let maxVal = logits[offset];
		for (let j = 1; j < CONFIG.vocabSize; j++) {
			if (logits[offset + j] > maxVal) {
				maxVal = logits[offset + j];
				maxIdx = j;
			}
		}

		sequence.push(maxIdx);
	}

	// Convert generated tokens to string (only valid tokens, exclude padding)
	const generatedText = String.fromCharCode(...sequence);
	console.log(`  Generated text: "${generatedText}"`);
	console.log(`  Token sequence (${sequence.length}): [${sequence.join(', ')}]`);

	const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
	console.log(`  Avg inference time: ${avgTime.toFixed(2)}ms`);
	console.log(`  Tokens/sec: ${(1000 / avgTime).toFixed(2)}`);
	console.log('  ✓ Generation test passed');
}


/**
 * Create random input tensor
 */
function createRandomInput(batchSize, seqLen) {
	const size = batchSize * seqLen;
	const data = new BigInt64Array(size);
	for (let i = 0; i < size; i++) {
		data[i] = BigInt(Math.floor(Math.random() * CONFIG.vocabSize));
	}
	return data;
}


/**
 * Validate output tensor
 */
function validateOutput(logits, batchSize, seqLen) {
	const expectedShape = [batchSize, seqLen, CONFIG.vocabSize];

	if (logits.dims.length !== 3) {
		throw new Error(`Expected 3D output, got ${logits.dims.length}D`);
	}

	if (logits.dims[0] !== expectedShape[0] ||
	    logits.dims[1] !== expectedShape[1] ||
	    logits.dims[2] !== expectedShape[2]) {
		throw new Error(
			`Shape mismatch! Expected [${expectedShape.join(', ')}], ` +
			`got [${logits.dims.join(', ')}]`
		);
	}

	if (logits.type !== 'float32') {
		throw new Error(`Expected float32 output, got ${logits.type}`);
	}
}


/**
 * Get predicted tokens from logits
 */
function getPredictions(logitsData, numPositions, vocabSize) {
	const predictions = [];
	for (let i = 0; i < numPositions; i++) {
		let maxIdx = 0;
		let maxVal = logitsData[i * vocabSize];
		for (let j = 1; j < vocabSize; j++) {
			const val = logitsData[i * vocabSize + j];
			if (val > maxVal) {
				maxVal = val;
				maxIdx = j;
			}
		}
		predictions.push(maxIdx);
	}
	return predictions;
}


/**
 * Main entry point
 */
async function main() {
	try {
		const success = await runTests();
		process.exit(success ? 0 : 1);
	} catch (error) {
		console.error('\n' + '='.repeat(80));
		console.error('FATAL ERROR');
		console.error('='.repeat(80));
		console.error(error);
		process.exit(1);
	}
}


// Run if called directly
if (require.main === module) {
	main();
}


module.exports = { runTests, testBasicInference, testInference, testGeneration };
