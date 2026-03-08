import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Shield, 
  AlertTriangle, 
  AlertCircle,
  CheckCircle2,
  Activity,
  TrendingUp,
  Stethoscope
} from 'lucide-react';
import { motion } from 'framer-motion';

const riskConfig = {
  low: {
    color: 'from-emerald-500 to-green-500',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    text: 'text-emerald-400',
    icon: CheckCircle2,
    label: 'Low Risk',
    recommendation: 'Continue routine prenatal care. Maintain healthy lifestyle habits and attend regular check-ups.',
  },
  mid: {
    color: 'from-amber-500 to-yellow-500',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    text: 'text-amber-400',
    icon: AlertTriangle,
    label: 'Medium Risk',
    recommendation: 'Enhanced monitoring recommended. Schedule more frequent prenatal visits and discuss risk factors with your healthcare provider.',
  },
  high: {
    color: 'from-rose-500 to-red-500',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/30',
    text: 'text-rose-400',
    icon: AlertCircle,
    label: 'High Risk',
    recommendation: 'Immediate clinical consultation recommended. Please contact your healthcare provider as soon as possible for comprehensive evaluation.',
  },
};

export default function ResultsPanel({ result, variant = 'default' }) {
  if (!result) return null;

  const config = riskConfig[result.riskLevel];
  const Icon = config.icon;
  const isPatient = variant === 'patient';

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-6"
    >
      {/* Main Risk Card */}
      <Card className={`
        border-2 overflow-hidden
        ${isPatient ? `bg-white ${config.border}` : `bg-slate-800/50 backdrop-blur-xl ${config.border}`}
      `}>
        <CardContent className="p-0">
          {/* Risk Level Header */}
          <div className={`bg-gradient-to-r ${config.color} p-6 text-white`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
                  <Icon className="w-8 h-8" />
                </div>
                <div>
                  <p className="text-white/80 text-sm font-medium">Predicted Risk Level</p>
                  <h2 className="text-3xl font-bold">{config.label}</h2>
                </div>
              </div>
              <div className="text-right">
                <p className="text-white/80 text-sm">Confidence</p>
                <p className="text-4xl font-bold">{result.confidence}%</p>
              </div>
            </div>
          </div>

          {/* Probability Breakdown */}
          <div className={`p-6 ${isPatient ? 'bg-slate-50' : 'bg-slate-800/30'}`}>
            <h3 className={`text-sm font-semibold mb-4 ${isPatient ? 'text-slate-700' : 'text-slate-300'}`}>
              Risk Probability Distribution
            </h3>
            <div className="space-y-3">
              {[
                { label: 'Low Risk', value: result.probabilities?.low || 25, color: 'bg-emerald-500' },
                { label: 'Medium Risk', value: result.probabilities?.mid || 35, color: 'bg-amber-500' },
                { label: 'High Risk', value: result.probabilities?.high || 40, color: 'bg-rose-500' },
              ].map((item) => (
                <div key={item.label} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className={isPatient ? 'text-slate-600' : 'text-slate-400'}>{item.label}</span>
                    <span className={`font-semibold ${isPatient ? 'text-slate-700' : 'text-white'}`}>{item.value}%</span>
                  </div>
                  <div className={`h-2 rounded-full ${isPatient ? 'bg-slate-200' : 'bg-slate-700'}`}>
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.value}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className={`h-full rounded-full ${item.color}`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendation Card */}
      <Card className={`
        border overflow-hidden
        ${isPatient 
          ? `bg-white ${config.border}` 
          : `bg-slate-800/50 backdrop-blur-xl border-white/10`}
      `}>
        <CardHeader className={`border-b ${isPatient ? 'border-slate-100' : 'border-white/10'}`}>
          <CardTitle className={`flex items-center gap-3 ${isPatient ? 'text-slate-800' : 'text-white'}`}>
            <div className={`w-10 h-10 rounded-xl ${config.bg} flex items-center justify-center`}>
              <Stethoscope className={`w-5 h-5 ${config.text}`} />
            </div>
            Clinical Recommendation
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <p className={`text-lg leading-relaxed ${isPatient ? 'text-slate-600' : 'text-slate-300'}`}>
            {config.recommendation}
          </p>
          <div className={`mt-4 flex items-center gap-2 text-sm ${isPatient ? 'text-slate-500' : 'text-slate-400'}`}>
            <Shield className="w-4 h-4" />
            <span>This assessment is for screening purposes only</span>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}