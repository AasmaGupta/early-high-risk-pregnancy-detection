import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import { Users, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';

const distributionData = [
  { range: '0-10%', count: 245, category: 'low' },
  { range: '10-20%', count: 312, category: 'low' },
  { range: '20-30%', count: 287, category: 'low' },
  { range: '30-40%', count: 198, category: 'mid' },
  { range: '40-50%', count: 156, category: 'mid' },
  { range: '50-60%', count: 134, category: 'mid' },
  { range: '60-70%', count: 98, category: 'high' },
  { range: '70-80%', count: 67, category: 'high' },
  { range: '80-90%', count: 42, category: 'high' },
  { range: '90-100%', count: 18, category: 'high' },
];

const categoryColors = {
  low: '#10B981',
  mid: '#F59E0B',
  high: '#EF4444',
};

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-xl">
        <p className="font-semibold text-white">Risk Score: {label}</p>
        <p className="text-sm text-slate-300">{payload[0].value} patients</p>
      </div>
    );
  }
  return null;
};

export default function PopulationContext({ currentRiskScore = 45 }) {
  // Calculate percentile
  const totalPatients = distributionData.reduce((sum, d) => sum + d.count, 0);
  let cumulativeCount = 0;
  let percentile = 0;
  
  for (const bucket of distributionData) {
    const bucketMax = parseInt(bucket.range.split('-')[1]);
    if (currentRiskScore <= bucketMax) {
      const bucketMin = parseInt(bucket.range.split('-')[0]);
      const positionInBucket = (currentRiskScore - bucketMin) / (bucketMax - bucketMin);
      percentile = Math.round(((cumulativeCount + bucket.count * positionInBucket) / totalPatients) * 100);
      break;
    }
    cumulativeCount += bucket.count;
  }

  const getPercentileInterpretation = (p) => {
    if (p <= 25) return { text: 'Lower risk than most patients', color: 'emerald' };
    if (p <= 50) return { text: 'Below average risk level', color: 'emerald' };
    if (p <= 75) return { text: 'Above average risk level', color: 'amber' };
    return { text: 'Higher risk than most patients', color: 'rose' };
  };

  const interpretation = getPercentileInterpretation(percentile);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
              <Users className="w-5 h-5 text-white" />
            </div>
            Population Risk Context
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {/* Percentile Summary */}
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30 text-center">
              <p className="text-sm text-slate-400 mb-1">Your Risk Percentile</p>
              <p className="text-4xl font-bold text-white">{percentile}<span className="text-lg text-slate-400">th</span></p>
              <p className={`text-sm mt-2 ${interpretation.color === 'emerald' ? 'text-emerald-400' : interpretation.color === 'amber' ? 'text-amber-400' : 'text-rose-400'}`}>
                {interpretation.text}
              </p>
            </div>
            <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
              <p className="text-sm text-slate-400 mb-2">Risk Score Comparison</p>
              <div className="flex items-end gap-2">
                <div className="flex-1">
                  <p className="text-xs text-slate-500">Your Score</p>
                  <p className="text-2xl font-bold text-teal-400">{currentRiskScore}%</p>
                </div>
                <div className="flex-1">
                  <p className="text-xs text-slate-500">Population Avg</p>
                  <p className="text-2xl font-bold text-slate-300">38%</p>
                </div>
              </div>
            </div>
          </div>

          {/* Distribution Chart */}
          <div className="mb-4">
            <h4 className="text-sm font-medium text-slate-400 mb-4">Historical Patient Risk Distribution (n={totalPatients.toLocaleString()})</h4>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="range" stroke="#94A3B8" fontSize={10} />
                  <YAxis stroke="#94A3B8" fontSize={10} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine 
                    x={distributionData.find(d => {
                      const max = parseInt(d.range.split('-')[1]);
                      const min = parseInt(d.range.split('-')[0]);
                      return currentRiskScore >= min && currentRiskScore <= max;
                    })?.range}
                    stroke="#14B8A6" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    label={{ value: 'You', position: 'top', fill: '#14B8A6', fontSize: 11 }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {distributionData.map((entry, index) => (
                      <Cell key={index} fill={categoryColors[entry.category]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Legend */}
          <div className="flex justify-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-emerald-500" />
              <span className="text-xs text-slate-400">Low Risk (0-30%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-amber-500" />
              <span className="text-xs text-slate-400">Medium Risk (30-60%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-rose-500" />
              <span className="text-xs text-slate-400">High Risk (60-100%)</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}