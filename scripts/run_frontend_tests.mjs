import { runAiBackendServiceTestCases } from 'file:///d:/AI_Check_Lac/src/services/aiBackendServiceTestCases.js';

async function main() {
  console.log('Running Frontend AI Backend Service Test Cases...');
  const result = await runAiBackendServiceTestCases();
  console.log(`Summary: ${result.passed} / ${result.total} PASSED\n`);
  result.results.forEach((r, idx) => {
    console.log(`[${r.passed ? 'PASS' : 'FAIL'}] ${r.name}`);
    if (!r.passed) {
      console.error(`       Error: ${r.error}`);
    }
  });

  if (result.passed !== result.total) {
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Fatal test runner error:', err);
  process.exit(1);
});
