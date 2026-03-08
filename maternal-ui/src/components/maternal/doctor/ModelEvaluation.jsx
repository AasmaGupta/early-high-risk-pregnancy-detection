import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';
import { CheckSquare, Target } from 'lucide-react';
import { motion } from 'framer-motion';

const confusionMatrix = [
  [142, 8, 5],    // True Low: predicted as Low, Mid, High
  [12, 98, 7],    // True Mid: predicted as Low, Mid, High
  [3, 9, 116],    // True High: predicted as Low, Mid, High
];

const classificationReport = [
  { class: 'Low Risk', precision: 0.90, recall: 0.92, f1: 0.91, support: 155 },
  { class: 'Mid Risk', precision: 0.85, recall: 0.84, f1: 0.84, support: 117 },
  { class: 'High Risk', precision: 0.91, recall: 0.91, f1: 0.91, support: 128 },
];

const probabilityDist = [
  { range: '0-20%', low: 145, mid: 12, high: 5 },
  { range: '20-40%', low: 8, mid: 28, high: 8 },
  { range: '40-60%', low: 2, mid: 58, high: 15 },
  { range: '60-80%', low: 0, mid: 15, high: 42 },
  { range: '80-100%', low: 0, mid: 4, high: 58 },
];

const COLORS = ['#10B981', '#F59E0B', '#EF4444'];

export default function ModelEvaluation() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Confusion Matrix */}
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-pink-500 flex items-center justify-center">
              <Target className="w-5 h-5 text-white" />
            </div>
            Confusion Matrix
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <p className="text-slate-400 text-sm mb-4">
            Shows how well the model classifies each risk category. Diagonal values (highlighted) indicate correct predictions — higher values mean better accuracy for that risk level.
          </p>
          <div className="grid grid-cols-4 gap-1 max-w-md mx-auto">
            <div className="col-span-1"></div>
            <div className="text-center text-xs text-slate-400 py-2">Low</div>
            <div className="text-center text-xs text-slate-400 py-2">Mid</div>
            <div className="text-center text-xs text-slate-400 py-2">High</div>
            
            {['Low', 'Mid', 'High'].map((label, rowIndex) => (
              <React.Fragment key={label}>
                <div className="text-right text-xs text-slate-400 pr-3 py-4">{label}</div>
                {confusionMatrix[rowIndex].map((value, colIndex) => {
                  const isCorrect = rowIndex === colIndex;
                  return (
                    <div
                      key={colIndex}
                      className={`
                        flex items-center justify-center py-4 rounded-lg font-mono font-semibold
                        ${isCorrect 
                          ? 'bg-teal-500/30 text-teal-400 border border-teal-500/50' 
                          : 'bg-slate-700/30 text-slate-400 border border-slate-600/30'}
                      `}
                    >
                      {value}
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
          <div className="flex justify-center gap-4 mt-4 text-xs text-slate-400">
            <span>Rows: Actual</span>
            <span>•</span>
            <span>Columns: Predicted</span>
          </div>
        </CardContent>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Classification Report */}
        <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
          <CardHeader className="border-b border-white/10">
            <CardTitle className="flex items-center gap-3 text-white text-lg">
              <CheckSquare className="w-5 h-5 text-teal-400" />
              Classification Report
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <p className="text-slate-400 text-sm mb-4">
              Precision = accuracy of positive predictions. Recall = ability to find all positive cases. F1 = balance of both. Higher scores indicate better model reliability.
            </p>
            <div className="space-y-4">
              {classificationReport.map((item, index) => (
                <div key={item.class} className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-white">{item.class}</span>
                    <Badge 
                      className={`${
                        index === 0 ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' :
                        index === 1 ? 'bg-amber-500/20 text-amber-400 border-amber-500/30' :
                        'bg-rose-500/20 text-rose-400 border-rose-500/30'
                      }`}
                    >
                      n={item.support}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-xs text-slate-400">Precision</p>
                      <p className="text-lg font-mono font-semibold text-white">{(item.precision * 100).toFixed(0)}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">Recall</p>
                      <p className="text-lg font-mono font-semibold text-white">{(item.recall * 100).toFixed(0)}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">F1 Score</p>
                      <p className="text-lg font-mono font-semibold text-teal-400">{(item.f1 * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Probability Distribution */}
        <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
          <CardHeader className="border-b border-white/10">
            <CardTitle className="text-white text-lg">
              Probability Distribution
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <p className="text-slate-400 text-sm mb-4">
              Shows how confident the model is when making predictions. Well-calibrated models show high probabilities for correct risk categories.
            </p>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={probabilityDist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="range" stroke="#94A3B8" fontSize={10} />
                  <YAxis stroke="#94A3B8" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1E293B', 
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="low" stackId="a" fill="#10B981" name="Low Risk" />
                  <Bar dataKey="mid" stackId="a" fill="#F59E0B" name="Mid Risk" />
                  <Bar dataKey="high" stackId="a" fill="#EF4444" name="High Risk" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-6 mt-4">
              {['Low', 'Mid', 'High'].map((label, index) => (
                <div key={label} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: COLORS[index] }} />
                  <span className="text-xs text-slate-400">{label} Risk</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}