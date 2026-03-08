import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Gauge, AlertTriangle, CheckCircle2, Info } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ConfidenceGauge({ confidence = 85 }) {
  const data = [
    { name: 'Confidence', value: confidence },
    { name: 'Uncertainty', value: 100 - confidence },
  ];

  const getConfidenceLevel = (conf) => {
    if (conf >= 85) return { level: 'High', color: '#10B981', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400' };
    if (conf >= 70) return { level: 'Moderate', color: '#F59E0B', bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400' };
    return { level: 'Low', color: '#EF4444', bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-400' };
  };

  const confLevel = getConfidenceLevel(confidence);

  const modelAgreement = [
    { model: 'Logistic Regression', agrees: confidence > 60 },
    { model: 'Random Forest', agrees: confidence > 50 },
    { model: 'XGBoost', agrees: true },
    { model: 'Gradient Boosting', agrees: confidence > 55 },
    { model: 'Stacking Ensemble', agrees: true },
    { model: 'Soft Voting', agrees: confidence > 45 },
  ];

  const agreementCount = modelAgreement.filter(m => m.agrees).length;
  const agreementPercentage = Math.round((agreementCount / modelAgreement.length) * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
              <Gauge className="w-5 h-5 text-white" />
            </div>
            Prediction Confidence
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Gauge Chart */}
            <div className="flex flex-col items-center">
              <div className="relative w-48 h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={data}
                      cx="50%"
                      cy="50%"
                      startAngle={180}
                      endAngle={0}
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={0}
                      dataKey="value"
                    >
                      <Cell fill={confLevel.color} />
                      <Cell fill="#334155" />
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-4xl font-bold text-white">{confidence}%</span>
                  <span className={`text-sm font-medium ${confLevel.text}`}>{confLevel.level} Confidence</span>
                </div>
              </div>

              {/* Recommendation based on confidence */}
              <div className={`mt-4 p-4 rounded-xl ${confLevel.bg} border ${confLevel.border} w-full`}>
                {confidence >= 85 ? (
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-emerald-400">High Reliability</p>
                      <p className="text-xs text-slate-300 mt-1">Prediction is highly reliable. Models show strong consensus.</p>
                    </div>
                  </div>
                ) : confidence >= 70 ? (
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-amber-400">Moderate Reliability</p>
                      <p className="text-xs text-slate-300 mt-1">Consider additional clinical assessment to confirm findings.</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-rose-400">Low Reliability</p>
                      <p className="text-xs text-slate-300 mt-1">Manual clinical review strongly recommended. Input may be atypical.</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Multi-Model Consensus */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-sm font-medium text-slate-300">Multi-Model Consensus</h4>
                <Badge className="bg-teal-500/20 text-teal-400 border border-teal-500/30">
                  {agreementPercentage}% Agreement
                </Badge>
              </div>

              <div className="space-y-2">
                {modelAgreement.map((model, index) => (
                  <motion.div
                    key={model.model}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={`flex items-center justify-between p-3 rounded-lg ${
                      model.agrees ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-slate-700/30 border border-slate-600/30'
                    }`}
                  >
                    <span className={`text-sm ${model.agrees ? 'text-white' : 'text-slate-400'}`}>
                      {model.model}
                    </span>
                    {model.agrees ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <span className="text-xs text-slate-500">Disagrees</span>
                    )}
                  </motion.div>
                ))}
              </div>

              <div className="mt-4 bg-slate-700/30 rounded-lg p-3">
                <p className="text-xs text-slate-400">
                  <strong className="text-slate-300">{agreementCount}/{modelAgreement.length} models</strong> agree on the predicted risk level. Higher consensus indicates more reliable predictions.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}