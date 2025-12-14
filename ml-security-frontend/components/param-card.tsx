import React from 'react';

const ParamCard = ({ label, value, highlight = false }: { label: string; value: string | number | undefined | null; highlight?: boolean }) => (
  <div className={`rounded-xl p-3 border transition-all ${
    highlight
      ? 'bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200'
      : 'bg-white/60 backdrop-blur-sm border-zinc-200/50'
  }`}>
    <div className="text-sm font-medium text-zinc-500 mb-1">{label}</div>
    <div className={`font-semibold ${highlight ? 'text-blue-900' : 'text-zinc-800'}`}>
      {value || "Not set"}
    </div>
  </div>
)

export default ParamCard;