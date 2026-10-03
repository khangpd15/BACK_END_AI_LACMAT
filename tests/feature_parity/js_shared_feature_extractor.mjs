import fs from 'node:fs';

const ALL_SHARED_FEATURES = [
  'leftValidRatio',
  'rightValidRatio',
  'bothValidRatio',
  'meanDeltaX',
  'medianDeltaX',
  'stdDeltaX',
  'minDeltaX',
  'maxDeltaX',
  'rangeDeltaX',
  'meanAbsDeltaX',
  'meanDeltaY',
  'medianDeltaY',
  'stdDeltaY',
  'minDeltaY',
  'maxDeltaY',
  'rangeDeltaY',
  'meanAbsDeltaY',
  'meanLeftX',
  'stdLeftX',
  'meanLeftY',
  'stdLeftY',
  'meanRightX',
  'stdRightX',
  'meanRightY',
  'stdRightY',
  'meanLeftVelocity',
  'peakLeftVelocity',
  'meanRightVelocity',
  'peakRightVelocity',
  'velocityDisparity',
];

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round((value + Number.EPSILON) * factor) / factor;
}

function mean(values) {
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function std(values) {
  const mu = mean(values);
  return Math.sqrt(mean(values.map((v) => (v - mu) ** 2)));
}

function stats(values) {
  return {
    mean: mean(values),
    median: median(values),
    std: std(values),
    min: Math.min(...values),
    max: Math.max(...values),
    range: Math.max(...values) - Math.min(...values),
    meanAbs: mean(values.map(Math.abs)),
  };
}

function extractSharedFeatures(samples) {
  if (samples.length === 0) {
    return Object.fromEntries(ALL_SHARED_FEATURES.map((name) => [name, null]));
  }

  const rawT = samples.map((s) => s.t);
  const tsSec = rawT.length > 1 && rawT.at(-1) - rawT[0] > 200
    ? rawT.map((t) => t / 1000)
    : rawT.map(Number);

  const left = [];
  const right = [];
  const both = [];

  samples.forEach((s, i) => {
    const lx = s.leftX;
    const ly = s.leftY;
    const rx = s.rightX;
    const ry = s.rightY;
    const lv = Boolean(s.leftValid && lx != null && ly != null && Number.isFinite(lx) && Number.isFinite(ly));
    const rv = Boolean(s.rightValid && rx != null && ry != null && Number.isFinite(rx) && Number.isFinite(ry));
    if (lv) left.push([tsSec[i], Number(lx), Number(ly)]);
    if (rv) right.push([tsSec[i], Number(rx), Number(ry)]);
    if (lv && rv) both.push([tsSec[i], Number(lx), Number(ly), Number(rx), Number(ry)]);
  });

  const result = {};
  result.leftValidRatio = round(left.length / samples.length, 4);
  result.rightValidRatio = round(right.length / samples.length, 4);
  result.bothValidRatio = round(both.length / samples.length, 4);

  if (both.length > 0) {
    const dx = both.map((r) => r[3] - r[1]);
    const dy = both.map((r) => r[4] - r[2]);
    const sx = stats(dx);
    const sy = stats(dy);
    result.meanDeltaX = round(sx.mean, 6);
    result.medianDeltaX = round(sx.median, 6);
    result.stdDeltaX = round(sx.std, 6);
    result.minDeltaX = round(sx.min, 6);
    result.maxDeltaX = round(sx.max, 6);
    result.rangeDeltaX = round(sx.range, 6);
    result.meanAbsDeltaX = round(sx.meanAbs, 6);
    result.meanDeltaY = round(sy.mean, 6);
    result.medianDeltaY = round(sy.median, 6);
    result.stdDeltaY = round(sy.std, 6);
    result.minDeltaY = round(sy.min, 6);
    result.maxDeltaY = round(sy.max, 6);
    result.rangeDeltaY = round(sy.range, 6);
    result.meanAbsDeltaY = round(sy.meanAbs, 6);
  }

  if (left.length > 0) {
    result.meanLeftX = round(mean(left.map((r) => r[1])), 6);
    result.stdLeftX = round(std(left.map((r) => r[1])), 6);
    result.meanLeftY = round(mean(left.map((r) => r[2])), 6);
    result.stdLeftY = round(std(left.map((r) => r[2])), 6);
  }

  if (right.length > 0) {
    result.meanRightX = round(mean(right.map((r) => r[1])), 6);
    result.stdRightX = round(std(right.map((r) => r[1])), 6);
    result.meanRightY = round(mean(right.map((r) => r[2])), 6);
    result.stdRightY = round(std(right.map((r) => r[2])), 6);
  }

  const leftV = [];
  for (let i = 0; i < left.length - 1; i += 1) {
    const dt = left[i + 1][0] - left[i][0];
    if (dt > 0.001) leftV.push(Math.hypot(left[i + 1][1] - left[i][1], left[i + 1][2] - left[i][2]) / dt);
  }
  const rightV = [];
  for (let i = 0; i < right.length - 1; i += 1) {
    const dt = right[i + 1][0] - right[i][0];
    if (dt > 0.001) rightV.push(Math.hypot(right[i + 1][1] - right[i][1], right[i + 1][2] - right[i][2]) / dt);
  }

  const meanLeftVelocity = leftV.length ? mean(leftV) : null;
  const meanRightVelocity = rightV.length ? mean(rightV) : null;
  result.meanLeftVelocity = meanLeftVelocity == null ? null : round(meanLeftVelocity, 6);
  result.peakLeftVelocity = leftV.length ? round(Math.max(...leftV), 6) : null;
  result.meanRightVelocity = meanRightVelocity == null ? null : round(meanRightVelocity, 6);
  result.peakRightVelocity = rightV.length ? round(Math.max(...rightV), 6) : null;
  result.velocityDisparity = meanLeftVelocity == null || meanRightVelocity == null
    ? null
    : round(Math.abs(meanRightVelocity - meanLeftVelocity), 6);

  for (const name of ALL_SHARED_FEATURES) {
    if (!(name in result)) result[name] = null;
  }
  return result;
}

const fixturePath = process.argv[2];
const samples = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
process.stdout.write(JSON.stringify(extractSharedFeatures(samples)));
