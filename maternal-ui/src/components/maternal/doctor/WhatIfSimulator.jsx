import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { 
  FlaskConical, 
  ArrowRight, 
  TrendingUp, 
  TrendingDown,
  Minus
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const simulateRisk = (data) => {
  const riskScore = 
    (data.systolicBP > 140 ? 25 : data.systolicBP > 130 ? 15 : data.systolicBP > 120 ? 5 : 0) +
    (data.diastolicBP > 90 ? 20 : data.diastolicBP > 85 ? 10 : 0) +
    (data.bloodSugar > 140 ? 25 : data.bloodSugar > 120 ? 15 : data.bloodSugar > 100 ? 5 : 0) +
    (data.age > 40 ? 15 : data.age > 35 ? 10 : data.age > 30 ? 5 : 0) +
    (data.heartRate > 100 ? 10 : data.heartRate > 90 ? 5 : 0) +
    (data.bodyTemp > 99.5 ? 5 : 0);

  let level = 'low';
  if (riskScore > 50) level = 'high';
  else if (riskScore > 25) level = 'mid';

  return { score: Math.min(100, riskScore + 15), level };
};

const parameters = [
  { key: 'systolicBP', label: 'Systolic BP', min: 70, max: 200, unit: 'mmHg' },
  { key: 'diastolicBP', label: 'Diastolic BP', min: 40, max: 130, unit: 'mmHg' },
  { key: 'bloodSugar', label: 'Blood Sugar', min: 50, max: 300, unit: 'mg/dL' },
  { key: 'heartRate', label: 'Heart Rate', min: 40, max: 120, unit: 'bpm' },
];

export default function WhatIfSimulator({ initialData }) {
  const [simData, setSimData] = useState({
    systolicBP: initialData?.systolicBP || 120,
    diastolicBP: initialData?.diastolicBP || 80,
    bloodSugar: initialData?.bloodSugar || 95,
    age: initialData?.age || 28,
    heartRate: initialData?.heartRate || 75,
    bodyTemp: initialData?.bodyTemp || 98.6,
  });

  const [baselineRisk, setBaselineRisk] = useState(null);
  const [currentRisk, setCurrentRisk] = useState(null);

  useEffect(() => {
    const baseline = simulateRisk(initialData || simData);
    setBaselineRisk(baseline);
    setCurrentRisk(baseline);
  }, []);

  useEffect(() => {
    const risk = simulateRisk(simData);
    setCurrentRisk(risk);
  }, [simData]);

  const handleChange = (key, value) => {
    setSimData(prev => ({ ...prev, [key]: value[0] }));
  };

  const riskDiff = currentRisk && baselineRisk ? currentRisk.score - baselineRisk.score : 0;

  const riskColors = {
    low: 'bg-emerald-500',
    mid: 'bg-amber-500',
    high: 'bg-rose-500',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <FlaskConical className="w-5 h-5 text-white" />
            </div>
            What-If Scenario Simulator
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Sliders */}
            <div className="space-y-6">
              <p className="text-sm text-slate-400">
                Adjust parameters to see how changes affect risk prediction in real-time.
              </p>
              {parameters.map((param) => (
                <div key={param.key} className="space-y-2">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium text-slate-300">{param.label}</label>
                    <span className="text-sm font-mono text-teal-400">{simData[param.key]} {param.unit}</span>
                  </div>
                  <Slider
                    value={[simData[param.key]]}
                    onValueChange={(value) => handleChange(param.key, value)}
                    min={param.min}
                    max={param.max}
                    className="[&_[role=slider]]:bg-white [&_[role=slider]]:border-white"
                  />
                </div>
              ))}
            </div>

            {/* Results */}
            <div className="space-y-4">
              {/* Risk Comparison */}
              <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-center flex-1">
                    <p className="text-xs text-slate-400 mb-1">Baseline Risk</p>
                    <p className="text-2xl font-bold text-slate-300">{baselineRisk?.score || 0}%</p>
                  </div>
                  <ArrowRight className="w-6 h-6 text-slate-500 mx-4" />
                  <div className="text-center flex-1">
                    <p className="text-xs text-slate-400 mb-1">Simulated Risk</p>
                    <p className={`text-2xl font-bold ${currentRisk?.level === 'high' ? 'text-rose-400' : currentRisk?.level === 'mid' ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {currentRisk?.score || 0}%
                    </p>
                  </div>
                </div>

                {/* Risk Change Indicator */}
                <div className={`flex items-center justify-center gap-2 p-3 rounded-lg ${riskDiff > 0 ? 'bg-rose-500/10' : riskDiff < 0 ? 'bg-emerald-500/10' : 'bg-slate-600/30'}`}>
                  {riskDiff > 0 ? (
                    <>
                      <TrendingUp className="w-5 h-5 text-rose-400" />
                      <span className="text-rose-400 font-semibold">+{riskDiff}% Risk Increase</span>
                    </>
                  ) : riskDiff < 0 ? (
                    <>
                      <TrendingDown className="w-5 h-5 text-emerald-400" />
                      <span className="text-emerald-400 font-semibold">{riskDiff}% Risk Decrease</span>
                    </>
                  ) : (
                    <>
                      <Minus className="w-5 h-5 text-slate-400" />
                      <span className="text-slate-400 font-semibold">No Change</span>
                    </>
                  )}
                </div>
              </div>

              {/* Risk Level Indicator */}
              <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
                <p className="text-sm text-slate-400 mb-3">Predicted Risk Level</p>
                <div className="flex gap-2">
                  {['low', 'mid', 'high'].map((level) => (
                    <div
                      key={level}
                      className={`flex-1 py-3 rounded-lg text-center font-semibold transition-all ${
                        currentRisk?.level === level
                          ? `${riskColors[level]} text-white shadow-lg`
                          : 'bg-slate-600/30 text-slate-500'
                      }`}
                    >
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </div>
                  ))}
                </div>
              </div>

              {/* Clinical Note */}
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3">
                <p className="text-xs text-amber-300">
                  <strong>Note:</strong> This simulator shows estimated risk changes. Actual clinical decisions should consider full patient history.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}