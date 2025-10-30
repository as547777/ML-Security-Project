import React from 'react';
import DefenseSelect from "@/components/defense-select";

const DefenseCard = () => {
  return (
    <div className="card-main flex-1 w-full max-w-2xl">
      <h2 className="text-xl font-semibold text-purple-700 mb-3">
        Defense Configuration
      </h2>
      <p className="text-zinc-600 mb-4">
        Select a defense strategy to mitigate adversarial attacks.
      </p>
      <DefenseSelect />
    </div>
  );
};

export default DefenseCard;