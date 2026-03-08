import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { Brain, TrendingUp, Layers } from 'lucide-react';
import { motion } from 'framer-motion';

const modelData = [
  { name: 'Logistic Reg', accuracy: 78.5, precision: 77.2, recall: 76.8, f1: 77.0, color: '#8B5CF6' },
  { name: 'Random Forest', accuracy: 84.3, precision: 83.5, recall: 84.1, f1: 83.8, color: '#10B981' },
  { name: 'XGBoost', accuracy: 86.2, precision: 85.4, recall: 86.0, f1: 85.7, color: '#F59E0B' },
  { name: 'Gradient Boost', accuracy: 85.1, precision: 84.3, recall: 84.9, f1: 84.6, color: '#EF4444' },
  { name: 'Stacking', accuracy: 87.19, precision: 86.5, recall: 86.8, f1: 86.6, color: '#06B6D4' },
  { name: 'Soft Voting', accuracy: 86.8, precision: 86.1, recall: 86.4, f1: 86.2, color: '#EC4899' },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-xl">
        <p className="font-semibold text-white mb-2">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm text-slate-300">
            {entry.name}: <span className="font-mono text-teal-400">{entry.value}%</span>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function ModelComparison() {
  const bestModel = modelData.reduce((a, b) => a.accuracy > b.accuracy ? a : b);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3 text-white">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                <Layers className="w-5 h-5 text-white" />
              </div>
              Model Performance Comparison
            </CardTitle>
            <Badge className="bg-teal-500/20 text-teal-400 border border-teal-500/30">
              Best: {bestModel.name} ({bestModel.accuracy.toFixed(2)}%)
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          <p className="text-slate-400 text-sm mb-4">
            Compares multiple ML algorithms to find the best predictor. <strong className="text-teal-400">Stacking</strong> combines predictions from all models using a meta-learner, achieving highest accuracy. <strong className="text-teal-400">Soft Voting</strong> averages probability outputs for robust predictions.
          </p>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94A3B8" fontSize={12} />
                <YAxis stroke="#94A3B8" fontSize={12} domain={[75, 100]} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
                  {modelData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
            {modelData.map((model, index) => (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: model.color }} />
                  <span className="font-medium text-white text-sm">{model.name}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p className="text-slate-400">Precision</p>
                    <p className="text-white font-mono">{model.precision}%</p>
                  </div>
                  <div>
                    <p className="text-slate-400">Recall</p>
                    <p className="text-white font-mono">{model.recall}%</p>
                  </div>
                  <div>
                    <p className="text-slate-400">F1 Score</p>
                    <p className="text-white font-mono">{model.f1}%</p>
                  </div>
                  <div>
                    <p className="text-slate-400">Accuracy</p>
                    <p className="text-teal-400 font-mono font-semibold">{model.accuracy}%</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}