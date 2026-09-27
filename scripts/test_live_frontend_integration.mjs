import { analyzeCoverTest } from 'file:///d:/AI_Check_Lac/src/services/aiBackendService.js';
import fs from 'fs';

async function testLiveIntegration() {
  console.log('Testing Live Frontend -> Backend AI Transfer Integration...');
  const sampleJson = JSON.parse(fs.readFileSync('d:/REMICARE-STRABISMUS-AI/data/raw/sample.json', 'utf8'));

  const result = await analyzeCoverTest(sampleJson);
  console.log('Result Status:', result.status);
  console.log('Prediction:', result.prediction);
  console.log('Class Probability:', result.classProbability);
  console.log('Domain Shift Warning:', result.domainShiftWarning);
  console.log('Clinical Meaning:', result.clinicalMeaning);
  console.log('Extracted Feature Count:', Object.keys(result.features || {}).length);

  if (result.status !== 'TRANSFER_EXPERIMENT') {
    throw new Error('Expected status TRANSFER_EXPERIMENT, got ' + result.status);
  }
  if (result.domainShiftWarning !== true) {
    throw new Error('Expected domainShiftWarning to be true');
  }
  if (result.clinicalMeaning !== null) {
    throw new Error('Expected clinicalMeaning to be null');
  }
  if (Object.keys(result.features || {}).length !== 30) {
    throw new Error('Expected exactly 30 features, got ' + Object.keys(result.features || {}).length);
  }

  console.log('\nSUCCESS: Live Frontend Service -> FastAPI integration verified!');
}

testLiveIntegration().catch(err => {
  console.error('Integration test failed:', err);
  process.exit(1);
});
