import React from 'react';
import AttackSelect from "@/components/attack-select";

const AttackCard = () => {
  return (
    <div className="card-main flex-1 w-full max-w-2xl">
      <h2 className="text-xl font-semibold text-blue-700 mb-3">
        Attack Configuration
      </h2>
      <p className="text-zinc-600 mb-4">
        Choose an adversarial attack method to test your model’s robustness.
      </p>
      <AttackSelect />
    </div>
  );
};

export default AttackCard;