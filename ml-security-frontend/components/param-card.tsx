import React, {ReactNode} from 'react';

const ParamCard = ({ className, label, value, highlight = false }: { className?: string; label: string; value: string | number | undefined | null | ReactNode; highlight?: boolean }) => {
  const displayValue = typeof value === 'number'
    ? parseFloat(value.toFixed(10)).toString()
    : (value ?? "Not set");

  return (
    <div className={`${className} rounded-xl p-3 border transition-all ${
      highlight
        ? 'bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200'
        : 'bg-white/60 backdrop-blur-sm border-zinc-200/50'
    }`}>
      <div className="text-sm font-medium text-zinc-500 mb-1">{label}</div>
      <div className={`font-semibold ${highlight ? 'text-blue-900' : 'text-zinc-800'}`}>
        {displayValue}
      </div>
    </div>
  );
}

export default ParamCard;