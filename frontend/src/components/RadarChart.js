const FEATURES = [
  'avg_sentence_length',
  'vocab_richness',
  'punctuation_density',
  'avg_word_length',
  'function_word_ratio',
  'lexical_density',
];

const LABELS = ['sentence len', 'vocab', 'punct.', 'word len', 'func. words', 'lex. density'];

const CX = 110, CY = 100, R = 80;

function angleFor(i) {
  return (Math.PI * 2 * i) / FEATURES.length - Math.PI / 2;
}

function toPoint(val, i) {
  const angle = angleFor(i);
  return [CX + val * R * Math.cos(angle), CY + val * R * Math.sin(angle)];
}

function polygon(values) {
  return values.map((v, i) => toPoint(v, i).join(',')).join(' ');
}

function gridPolygon(scale) {
  return FEATURES.map((_, i) => toPoint(scale, i).join(',')).join(' ');
}

export default function RadarChart({ userFeatures, authorFeatures }) {
  const userVals   = FEATURES.map(f => userFeatures[f]   ?? 0);
  const authorVals = FEATURES.map(f => authorFeatures[f] ?? 0);

  return (
    <svg width="100%" viewBox="0 0 220 200" style={{ display: 'block' }}>
      <g transform={`translate(0,0)`}>
        {/* Grid rings */}
        {[0.33, 0.66, 1].map(s => (
          <polygon key={s} points={gridPolygon(s)}
            fill="none" stroke="#D8D0BC" strokeWidth="0.5" opacity="0.6" />
        ))}
        {/* Axis lines */}
        {FEATURES.map((_, i) => {
          const [x, y] = toPoint(1, i);
          return <line key={i} x1={CX} y1={CY} x2={x} y2={y}
            stroke="#E0D8C8" strokeWidth="0.4" opacity="0.5" />;
        })}
        {/* Author polygon */}
        <polygon points={polygon(authorVals)}
          fill="#2E7D64" fillOpacity="0.1"
          stroke="#2E7D64" strokeWidth="1" strokeOpacity="0.4" />
        {/* User polygon */}
        <polygon points={polygon(userVals)}
          fill="#8B5E3C" fillOpacity="0.12"
          stroke="#8B5E3C" strokeWidth="1.2" strokeOpacity="0.6" />
        {/* Labels */}
        {FEATURES.map((_, i) => {
          const [x, y] = toPoint(1.18, i);
          return (
            <text key={i} x={x} y={y}
              textAnchor={x < CX - 5 ? 'end' : x > CX + 5 ? 'start' : 'middle'}
              fontSize="8" fill="#9A9080" fontFamily="EB Garamond,serif">
              {LABELS[i]}
            </text>
          );
        })}
      </g>
    </svg>
  );
}
