import { useState } from 'react';

export default function TextInput({ onSubmit, loading }) {
  const [text, setText] = useState('');
  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;

  function handleSubmit() {
    if (wordCount >= 100) onSubmit(text);
  }

  return (
    <div className="screen inp active">
      <div className="inp-texture" />
      <div className="inp-glow" />
      <div className="inp-inner">
        <div className="inp-top">
          <div className="inp-logo">QuillMatch</div>
          <div className="inp-tagline">
            Authorship · Style · Identity<br />
            Project Gutenberg · scikit-learn
          </div>
        </div>

        <div className="inp-hero">
          <div className="inp-eyebrow">Literary fingerprint engine</div>
          <div className="inp-title">
            Paste your words.<br />
            Meet your <em>author twin.</em>
          </div>
          <div className="inp-title-sub">
            We measure 11 dimensions of your writing style<br />
            and match you to a literary giant.
          </div>
        </div>

        <div className="inp-divider">
          <div className="inp-divider-line" />
          <div className="inp-divider-mark">✦ ✦ ✦</div>
          <div className="inp-divider-line" />
        </div>

        <div className="inp-field-wrap">
          <div className="inp-field-label">Your writing sample</div>
          <textarea
            className="inp-textarea"
            placeholder="Any prose — a story opening, an email, a journal entry. The algorithm reads rhythm, not meaning…"
            value={text}
            onChange={e => setText(e.target.value)}
          />
        </div>

        <div className="inp-row">
          <div className="inp-hint">
            {wordCount < 100
              ? `Minimum 100 words — ${wordCount} so far`
              : `${wordCount} words ✓`}
          </div>
          <button
            className="inp-btn"
            onClick={handleSubmit}
            disabled={wordCount < 100 || loading}
          >
            {loading ? 'Analysing…' : 'Analyse my style →'}
          </button>
        </div>

        <div className="inp-footer-stats">
          <div className="inp-stat">
            <div className="inp-stat-num">7</div>
            <div className="inp-stat-lbl">Authors</div>
          </div>
          <div className="inp-stat">
            <div className="inp-stat-num">11</div>
            <div className="inp-stat-lbl">Dimensions</div>
          </div>
          <div className="inp-stat">
            <div className="inp-stat-num">87%</div>
            <div className="inp-stat-lbl">CV accuracy</div>
          </div>
        </div>
      </div>
    </div>
  );
}
