export default function FeatureBar({ name, userVal, authorVal }) {
  return (
    <div className="feat-row">
      <div className="feat-head">
        <span className="feat-name">{name}</span>
        <span className="feat-vals">{userVal.toFixed(2)} · {authorVal.toFixed(2)}</span>
      </div>
      <div className="feat-track">
        <div className="feat-bar-a" style={{ width: `${authorVal * 100}%` }} />
        <div className="feat-bar-u" style={{ width: `${userVal * 100}%` }} />
      </div>
    </div>
  );
}
