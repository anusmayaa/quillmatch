import { useState } from 'react';
import './App.css';
import TextInput  from './components/TextInput';
import ResultCard from './components/ResultCard';
import { analyseText } from './api/api';

export default function App() {
  const [screen,  setScreen]  = useState('input');
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  async function handleSubmit(text) {
    setLoading(true);
    setError(null);
    try {
      const data = await analyseText(text);
      setResult(data);
      setScreen('result');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleBack() {
    setScreen('input');
    setResult(null);
  }

  return (
    <div className="qm-wrap">
      <div className="qm">
        {error && (
          <div className="qm-error">{error}</div>
        )}
        {screen === 'input' && (
          <TextInput onSubmit={handleSubmit} loading={loading} />
        )}
        {screen === 'result' && (
          <ResultCard result={result} onBack={handleBack} />
        )}
      </div>
    </div>
  );
}
