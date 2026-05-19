import { useEffect, useRef } from 'react';
import RadarChart from './RadarChart';
import FeatureBar from './FeatureBar';

const FEATURE_LABELS = {
  avg_sentence_length:      'Avg sentence length',
  vocab_richness:           'Vocab richness',
  punctuation_density:      'Punctuation density',
  avg_word_length:          'Avg word length',
  function_word_ratio:      'Function word ratio',
  sentence_length_variance: 'Sentence variance',
  comma_frequency:          'Comma frequency',
  question_mark_frequency:  'Question marks',
  exclamation_frequency:    'Exclamations',
  paragraph_length:         'Paragraph length',
  lexical_density:          'Lexical density',
};

const AUTHOR_INITIALS = {
  'Jane Austen':        'JA',
  'Charles Dickens':    'CD',
  'Mark Twain':         'MT',
  'Oscar Wilde':        'OW',
  'Virginia Woolf':     'VW',
  'Edgar Allan Poe':    'EP',
  'Arthur Conan Doyle': 'AD',
};

const AUTHOR_DESC = {
  'Jane Austen':        'English novelist · 1775 – 1817',
  'Charles Dickens':    'English novelist · 1812 – 1870',
  'Mark Twain':         'American author · 1835 – 1910',
  'Oscar Wilde':        'Irish playwright · 1854 – 1900',
  'Virginia Woolf':     'English writer · 1882 – 1941',
  'Edgar Allan Poe':    'American writer · 1809 – 1849',
  'Arthur Conan Doyle': 'Scottish author · 1859 – 1930',
};

export default function ResultCard({ result, onBack }) {
  const fillRef = useRef(null);
  const pctRef  = useRef(null);

  useEffect(() => {
    if (!result) return;
    const target = result.confidence;
    setTimeout(() => {
      if (fillRef.current) fillRef.current.style.width = `${target}%`;
      let v = 0;
      const iv = setInterval(() => {
        v = Math.min(v + 2, target);
        if (pctRef.current) pctRef.current.textContent = `${Math.round(v)}%`;
        if (v >= target) clearInterval(iv);
      }, 16);
    }, 100);
  }, [result]);

  if (!result) return null;

  const sortedScores = Object.entries(result.all_scores)
    .sort((a, b) => b[1] - a[1]);

  const featureEntries = Object.entries(result.user_features);

  return (
    <div className="screen res active">
      <div className="res-hero">
        <div className="res-hero-texture" />
        <div className="res-hero-inner">
          <div className="res-nav">
            <div className="res-back" onClick={onBack}>← analyse another</div>
            <div className="res-logo">QuillMatch</div>
          </div>

          <div className="res-reveal">
            <div className="res-avatar">
              {AUTHOR_INITIALS[result.author] ?? result.author[0]}
            </div>
            <div className="res-name-block">
              <div className="res-matched-lbl">Your literary twin</div>
              <div className="res-name">{result.author}</div>
              <div className="res-name-desc">
                {AUTHOR_DESC[result.author] ?? ''}
              </div>
            </div>
          </div>

          <div className="res-conf">
            <div className="res-conf-head">
              <div className="res-conf-lbl">Confidence</div>
              <div className="res-conf-pct" ref={pctRef}>0%</div>
            </div>
            <div className="res-conf-track">
              <div className="res-conf-fill" ref={fillRef} />
            </div>
          </div>

          <div className="res-scores">
            {sortedScores.map(([author, score]) => (
              <div
                key={author}
                className={`res-score-pill${author === result.author ? ' top' : ''}`}
              >
                {author.split(' ').pop()} {score}%
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="res-body">
        <div className="res-ornament-row">
          <div className="res-ornament-line" />
          <div className="res-ornament-text">✦</div>
          <div className="res-ornament-line" />
        </div>

        <div className="res-quote">
          <div className="res-quote-bigmark">"</div>
          <div className="res-quote-text">{result.explanation}</div>
          <div className="res-quote-rule" />
        </div>

        <div className="res-ornament-row">
          <div className="res-ornament-line" />
          <div className="res-ornament-text">✦</div>
          <div className="res-ornament-line" />
        </div>

        <div className="res-grid">
          <div>
            <div className="res-section-lbl">Feature breakdown</div>
            <div className="res-legend">
              <div className="res-leg">
                <div className="res-leg-dot" style={{ background: '#8B5E3C' }} />
                You
              </div>
              <div className="res-leg">
                <div className="res-leg-dot" style={{ background: '#2E7D64' }} />
                {result.author.split(' ')[1] ?? result.author}
              </div>
            </div>
            {featureEntries.map(([key, userVal]) => (
              <FeatureBar
                key={key}
                name={FEATURE_LABELS[key] ?? key}
                userVal={userVal}
                authorVal={result.author_features[key] ?? 0}
              />
            ))}
          </div>

          <div>
            <div className="res-section-lbl">Style radar</div>
            <RadarChart
              userFeatures={result.user_features}
              authorFeatures={result.author_features}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
