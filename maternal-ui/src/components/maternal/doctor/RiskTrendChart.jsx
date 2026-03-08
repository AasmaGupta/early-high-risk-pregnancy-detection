import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, AreaChart
} from 'recharts';
import { TrendingUp, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

const trendData = [
  { date: 'Week 1', risk: 25, systolic: 118, diastolic: 76, sugar: 92, heartRate: 74 },
  { date: 'Week 2', risk: 28, systolic: 122, diastolic: 78, sugar: 95, heartRate: 76 },
  { date: 'Week 3', risk: 32, systolic: 126, diastolic: 82, sugar: 102, heartRate: 78 },
  { date: 'Week 4', risk: 35, systolic: 130, diastolic: 84, sugar: 108, heartRate: 80 },
  { date: 'Week 5', risk: 42, systolic: 138, diastolic: 88, sugar: 118, heartRate: 84 },
  { date: 'Week 6', risk: 38, systolic: 132, diastolic: 85, sugar: 112, heartRate: 82 },
  { date: 'Week 7', risk: 45, systolic: 142, diastolic: 90, sugar: 125, heartRate: 86 },
  { date: 'Week 8', risk: 52, systolic: 148, diastolic: 94, sugar: 135, heartRate: 88 },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 shadow-xl">
        <p className="font-semibold text-white mb-2">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {entry.value}{entry.name === 'Risk Score' ? '%' : ''}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function RiskTrendChart() {
  const latestRisk = trendData[trendData.length - 1].risk;
  const previousRisk = trendData[trendData.length - 2].risk;
  const trend = latestRisk - previousRisk;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3 text-white">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-500 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              Risk Trend Analysis
            </CardTitle>
            <Badge className={`${trend > 0 ? 'bg-rose-500/20 text-rose-400 border-rose-500/30' : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'}`}>
              {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from last week
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          {/* Risk Score Trend */}
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-400 mb-4">Risk Score Progression</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData}>
                  <defs>
                    <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" stroke="#94A3B8" fontSize={11} />
                  <YAxis stroke="#94A3B8" fontSize={11} domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area 
                    type="monotone" 
                    dataKey="risk" 
                    name="Risk Score"
                    stroke="#EF4444" 
                    strokeWidth={3}
                    fill="url(#riskGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Vital Signs Trends */}
          <div>
            <h4 className="text-sm font-medium text-slate-400 mb-4">Vital Signs Trends</h4>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" stroke="#94A3B8" fontSize={11} />
                  <YAxis stroke="#94A3B8" fontSize={11} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line type="monotone" dataKey="systolic" name="Systolic BP" stroke="#EF4444" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="diastolic" name="Diastolic BP" stroke="#F59E0B" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="sugar" name="Blood Sugar" stroke="#8B5CF6" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="heartRate" name="Heart Rate" stroke="#EC4899" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}