/**
 * Node.js Test Suite for Evaluation Mode ONNX Models
 *
 * Tests inference with EvaluationLM exported models that accept:
 * - input_ids: [batch, seq_len]
 *
 * Returns:
 * - values: [batch] - Predicted game outcome values in range [-1, 1]
 */

const ort = require('onnxruntime-node');
const path = require('path');
const fs = require('fs');

// Configuration
const CONFIG = {
	// Evaluation mode model
	modelPath: path.resolve(__dirname, '../../outputs/trigor/20251130-trigo-value-gpt2-l6-h64-251125-lr2000/GPT2CausalLM_ep0042_evaluation.onnx'),
	vocabSize: 128,  // Compact tokenizer vocab size

	// Special tokens (matching Python tokenizer)
	PAD_ID: 0,
	START_ID: 1,
	END_ID: 2,
	VALUE_ID: 3,

	// Default TGN file
	defaultTgnPath: path.resolve(__dirname, '../../../trigo/trigo-web/tools/output/selfplay/game_03b761486ee63865.tgn'),
};

/**
 * Simple TGN tokenizer matching Python implementation.
 * Uses direct ASCII mapping: token_id = ascii_value
 */
class TGNTokenizer {
	constructor() {
		this.VOCAB_SIZE = 128;
		this.PAD_ID = CONFIG.PAD_ID;
		this.START_ID = CONFIG.START_ID;
		this.END_ID = CONFIG.END_ID;
		this.VALUE_ID = CONFIG.VALUE_ID;
	}

	/**
	 * Encode text to token IDs.
	 * @param {string} text - Input text
	 * @param {Object} options - Encoding options
	 * @returns {BigInt64Array} - Token IDs
	 */
	encode(text, options = {}) {
		const {
			maxLength = 2048,
			addSpecialTokens = true,
			padding = false,
			truncation = true,
		} = options;

		// Convert text to bytes (UTF-8)
		const bytes = Buffer.from(text, 'utf8');

		// Map bytes to token IDs (direct ASCII mapping)
		const tokens = [];

		// Add START token if requested
		if (addSpecialTokens) {
			tokens.push(this.START_ID);
		}

		// Convert bytes to token IDs
		for (const byte of bytes) {
			// Only accept valid tokens (0-127)
			if (byte < 128) {
				tokens.push(byte);
			}
			// Non-ASCII bytes are skipped (TGN is ASCII-only)
		}

		// Add END token if requested
		if (addSpecialTokens) {
			tokens.push(this.END_ID);
		}

		// Truncate if needed (preserve END token)
		if (truncation && tokens.length > maxLength) {
			tokens.length = maxLength - 1;
			tokens.push(this.END_ID);
		}

		// Pad if requested
		if (padding && tokens.length < maxLength) {
			while (tokens.length < maxLength) {
				tokens.push(this.PAD_ID);
			}
		}

		// Convert to BigInt64Array for ONNX
		return new BigInt64Array(tokens.map(t => BigInt(t)));
	}

	/**
	 * Decode token IDs to text.
	 * @param {Array|BigInt64Array} tokens - Token IDs
	 * @param {boolean} skipSpecialTokens - Skip special tokens (0-7)
	 * @returns {string} - Decoded text
	 */
	decode(tokens, skipSpecialTokens = true) {
		const bytes = [];

		for (const token of tokens) {
			const tokenNum = Number(token);

			// Skip special tokens if requested
			if (skipSpecialTokens && tokenNum < 8) {
				continue;
			}

			// Skip padding
			if (tokenNum === this.PAD_ID) {
				continue;
			}

			// Convert token to byte (direct mapping)
			if (tokenNum < 128) {
				bytes.push(tokenNum);
			}
		}

		// Convert bytes to UTF-8 string
		return Buffer.from(bytes).toString('utf8');
	}
}

/**
 * Load and preprocess TGN file.
 * Removes comments (lines starting with ';') from the end.
 * @param {string} tgnPath - Path to TGN file
 * @returns {string} - Cleaned TGN text
 */
function loadTGN(tgnPath) {
	if (!fs.existsSync(tgnPath)) {
		throw new Error(`TGN file not found: ${tgnPath}`);
	}

	const content = fs.readFileSync(tgnPath, 'utf8');

	// Split into lines
	const lines = content.split('\n');

	// Remove trailing empty lines and comments
	while (lines.length > 0) {
		const lastLine = lines[lines.length - 1].trim();
		// Remove if empty or starts with ';' (comment)
		if (lastLine === '' || lastLine.startsWith(';')) {
			lines.pop();
		} else {
			break;
		}
	}

	// Rejoin lines
	return lines.join('\n');
}

/**
 * Test basic value prediction on a TGN file.
 */
async function testValuePrediction(session, tgnPath) {
	console.log('\n' + '='.repeat(80));
	console.log('TEST: Value Prediction on TGN Game');
	console.log('='.repeat(80));

	// Load and clean TGN
	const tgnText = loadTGN(tgnPath);
	console.log(`\nTGN file: ${path.basename(tgnPath)}`);
	console.log(`TGN content (${tgnText.length} chars):`);
	console.log('-'.repeat(40));
	console.log(tgnText);
	console.log('-'.repeat(40));

	// Tokenize
	const tokenizer = new TGNTokenizer();
	const inputIds = tokenizer.encode(tgnText, {
		maxLength: 256,  // Match export seq_len (model adds +1 for VALUE token)
		addSpecialTokens: true,
		padding: true,  // Enable padding for fixed-size models
		truncation: true,
	});

	console.log(`\nTokenized sequence length: ${inputIds.length}`);
	console.log(`First 20 tokens: [${Array.from(inputIds.slice(0, 20)).map(t => Number(t)).join(', ')}]`);

	// Create ONNX tensor
	const inputTensor = new ort.Tensor('int64', inputIds, [1, inputIds.length]);

	// Run inference
	console.log(`\nRunning inference...`);
	const startTime = Date.now();
	const results = await session.run({
		input_ids: inputTensor
	});
	const inferenceTime = Date.now() - startTime;

	const values = results.values;
	const predictedValue = values.data[0];

	console.log(`\n${'='.repeat(80)}`);
	console.log(`RESULT`);
	console.log(`${'='.repeat(80)}`);
	console.log(`  Inference time: ${inferenceTime}ms`);
	console.log(`  Output shape: [${values.dims.join(', ')}]`);
	console.log(`  Predicted value: ${predictedValue.toFixed(6)}`);
	console.log(`  Value range: [-1.0, 1.0]`);
	console.log(`  Interpretation:`);
	if (predictedValue > 0.5) {
		console.log(`    Strong win for Black (+${predictedValue.toFixed(3)})`);
	} else if (predictedValue > 0) {
		console.log(`    Slight advantage for Black (+${predictedValue.toFixed(3)})`);
	} else if (predictedValue > -0.5) {
		console.log(`    Slight advantage for White (${predictedValue.toFixed(3)})`);
	} else {
		console.log(`    Strong win for White (${predictedValue.toFixed(3)})`);
	}
	console.log(`${'='.repeat(80)}`);

	// Validate
	const shapeMatch = values.dims.length === 1 && values.dims[0] === 1;
	const valueInRange = predictedValue >= -1.0 && predictedValue <= 1.0;

	if (shapeMatch && valueInRange) {
		console.log('\n✓ Test passed');
		return true;
	} else {
		console.log('\n✗ Test failed');
		if (!shapeMatch) console.log(`  - Shape mismatch: expected [1], got [${values.dims.join(', ')}]`);
		if (!valueInRange) console.log(`  - Value out of range: ${predictedValue}`);
		return false;
	}
}

/**
 * Test batch value prediction on multiple TGN files.
 */
async function testBatchPrediction(session, tgnPaths) {
	console.log('\n' + '='.repeat(80));
	console.log('TEST: Batch Value Prediction');
	console.log('='.repeat(80));

	const tokenizer = new TGNTokenizer();
	const batchSize = tgnPaths.length;

	console.log(`\nEvaluating ${batchSize} games...`);

	// Load and tokenize all TGN files with fixed length
	const FIXED_LEN = 256;  // Match export seq_len
	const allInputIds = [];

	for (const tgnPath of tgnPaths) {
		const tgnText = loadTGN(tgnPath);
		const inputIds = tokenizer.encode(tgnText, {
			maxLength: FIXED_LEN,
			addSpecialTokens: true,
			padding: true,  // Pad to fixed length
			truncation: true,
		});
		allInputIds.push(inputIds);
	}

	// Concatenate all sequences into batch tensor
	const paddedInputIds = new BigInt64Array(batchSize * FIXED_LEN);
	for (let i = 0; i < batchSize; i++) {
		const inputIds = allInputIds[i];
		for (let j = 0; j < inputIds.length; j++) {
			paddedInputIds[i * FIXED_LEN + j] = inputIds[j];
		}
	}

	// Create ONNX tensor
	const inputTensor = new ort.Tensor('int64', paddedInputIds, [batchSize, FIXED_LEN]);

	// Run inference
	const startTime = Date.now();
	const results = await session.run({
		input_ids: inputTensor
	});
	const inferenceTime = Date.now() - startTime;

	const values = results.values;

	console.log(`\nResults:`);
	console.log(`  Inference time: ${inferenceTime}ms (${(inferenceTime / batchSize).toFixed(2)}ms per game)`);
	console.log(`  Output shape: [${values.dims.join(', ')}]`);
	console.log(`\n  Game values:`);

	for (let i = 0; i < batchSize; i++) {
		const value = values.data[i];
		console.log(`    ${path.basename(tgnPaths[i])}: ${value.toFixed(6)}`);
	}

	// Validate
	const shapeMatch = values.dims[0] === batchSize;
	const allInRange = Array.from(values.data).every(v => v >= -1.0 && v <= 1.0);

	if (shapeMatch && allInRange) {
		console.log('\n✓ Batch test passed');
		return true;
	} else {
		console.log('\n✗ Batch test failed');
		return false;
	}
}

/**
 * Run all tests.
 */
async function runTests() {
	console.log('\n' + '='.repeat(80));
	console.log('ONNX Evaluation Mode Inference Test Suite (Node.js)');
	console.log('='.repeat(80));

	// Check if model file exists
	if (!fs.existsSync(CONFIG.modelPath)) {
		console.error(`\n✗ Model file not found: ${CONFIG.modelPath}`);
		console.error('\nTo generate the model, run:');
		console.error('  python exportOnnx.py <training_dir> --evaluation-mode --checkpoint best\n');
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

	// Test 1: Single TGN file
	if (fs.existsSync(CONFIG.defaultTgnPath)) {
		const passed = await testValuePrediction(session, CONFIG.defaultTgnPath);
		if (passed) results.passed++; else results.failed++;
	} else {
		console.log(`\nℹ Default TGN file not found: ${CONFIG.defaultTgnPath}`);
		console.log('Skipping single file test');
	}

	// Test 2: Batch prediction (if multiple TGN files available)
	const tgnDir = path.dirname(CONFIG.defaultTgnPath);
	if (fs.existsSync(tgnDir)) {
		const tgnFiles = fs.readdirSync(tgnDir)
			.filter(f => f.endsWith('.tgn'))
			.map(f => path.join(tgnDir, f))
			.slice(0, 3);  // Test with first 3 files

		if (tgnFiles.length > 1) {
			const passed = await testBatchPrediction(session, tgnFiles);
			if (passed) results.passed++; else results.failed++;
		}
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
