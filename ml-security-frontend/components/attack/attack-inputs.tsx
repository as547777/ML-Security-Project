"use client"
import {useData} from "@/context/DataContext";
import ParamInputs from "@/components/param-inputs";

const AttackInputs = () => {
  const { attackParams, updateAttackParams } = useData()

  return <ParamInputs params={attackParams} updateParams={updateAttackParams} />
};

export default AttackInputs;