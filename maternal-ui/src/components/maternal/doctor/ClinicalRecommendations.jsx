import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Stethoscope, 
  Calendar, 
  Pill, 
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock,
  FileText
} from 'lucide-react';
import { motion } from 'framer-motion';

const recommendations = {
  low: {
    title: 'Low Risk - Routine Care',
    color: 'emerald',
    icon: CheckCircle2,
    summary: 'Continue standard prenatal care protocol with regular monitoring.',
    actions: [
      { icon: Calendar, text: 'Schedule routine prenatal visits every 4 weeks', priority: 'standard' },
      { icon: Activity, text: 'Maintain regular physical activity as advised', priority: 'standard' },
      { icon: Pill, text: 'Continue prenatal vitamins and supplements', priority: 'standard' },
      { icon: FileText, text: 'Complete standard screening tests per protocol', priority: 'standard' },
    ],
    monitoring: 'Standard monitoring frequency. No immediate interventions required.',
    followUp: '4 weeks',
  },
  mid: {
    title: 'Medium Risk - Enhanced Monitoring',
    color: 'amber',
    icon: AlertCircle,
    summary: 'Increased monitoring recommended. Address modifiable risk factors.',
    actions: [
      { icon: Calendar, text: 'Increase prenatal visit frequency to every 2 weeks', priority: 'elevated' },
      { icon: Activity, text: 'Monitor blood pressure at home daily', priority: 'elevated' },
      { icon: Pill, text: 'Review medication and supplement regimen', priority: 'elevated' },
      { icon: Stethoscope, text: 'Consider specialist consultation if symptoms develop', priority: 'elevated' },
      { icon: FileText, text: 'Order additional diagnostic tests as indicated', priority: 'standard' },
    ],
    monitoring: 'Enhanced monitoring with home BP tracking. Weekly review of vitals.',
    followUp: '2 weeks',
  },
  high: {
    title: 'High Risk - Immediate Attention',
    color: 'rose',
    icon: AlertCircle,
    summary: 'Immediate clinical evaluation required. Consider specialist referral.',
    actions: [
      { icon: Stethoscope, text: 'Urgent consultation with maternal-fetal medicine specialist', priority: 'urgent' },
      { icon: Calendar, text: 'Weekly or more frequent prenatal monitoring', priority: 'urgent' },
      { icon: Activity, text: 'Continuous blood pressure and glucose monitoring', priority: 'urgent' },
      { icon: Pill, text: 'Evaluate need for antihypertensive or other medications', priority: 'elevated' },
      { icon: FileText, text: 'Comprehensive diagnostic workup including ultrasound', priority: 'urgent' },
      { icon: Clock, text: 'Discuss delivery timing and birth plan', priority: 'elevated' },
    ],
    monitoring: 'Intensive monitoring protocol. Consider hospitalization if conditions worsen.',
    followUp: '1 week or sooner',
  },
};

const priorityStyles = {
  urgent: 'bg-rose-500/20 border-rose-500/30 text-rose-400',
  elevated: 'bg-amber-500/20 border-amber-500/30 text-amber-400',
  standard: 'bg-slate-600/30 border-slate-500/30 text-slate-300',
};

export default function ClinicalRecommendations({ riskLevel = 'low' }) {
  const rec = recommendations[riskLevel] || recommendations.low;
  const colorMap = {
    emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', gradient: 'from-emerald-500 to-green-500' },
    amber: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', gradient: 'from-amber-500 to-orange-500' },
    rose: { bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-400', gradient: 'from-rose-500 to-red-500' },
  };
  const colors = colorMap[rec.color];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className={`border-b border-white/10 ${colors.bg}`}>
          <CardTitle className="flex items-center gap-3 text-white">
            <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${colors.gradient} flex items-center justify-center`}>
              <Stethoscope className="w-5 h-5 text-white" />
            </div>
            Clinical Recommendations
            <Badge className={`ml-auto ${colors.bg} ${colors.text} border ${colors.border}`}>
              {rec.title}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {/* Summary */}
          <div className={`p-4 rounded-xl ${colors.bg} border ${colors.border} mb-6`}>
            <div className="flex items-start gap-3">
              <rec.icon className={`w-5 h-5 ${colors.text} flex-shrink-0 mt-0.5`} />
              <p className={`${colors.text} font-medium`}>{rec.summary}</p>
            </div>
          </div>

          {/* Action Items */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-slate-300 mb-4">Recommended Actions</h4>
            <div className="space-y-3">
              {rec.actions.map((action, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`flex items-center gap-3 p-3 rounded-lg border ${priorityStyles[action.priority]}`}
                >
                  <action.icon className="w-5 h-5 flex-shrink-0" />
                  <span className="text-sm flex-1">{action.text}</span>
                  <Badge variant="outline" className="text-xs capitalize">
                    {action.priority}
                  </Badge>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Monitoring & Follow-up */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-teal-400" />
                <h5 className="font-medium text-white text-sm">Monitoring Protocol</h5>
              </div>
              <p className="text-xs text-slate-400">{rec.monitoring}</p>
            </div>
            <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-teal-400" />
                <h5 className="font-medium text-white text-sm">Recommended Follow-up</h5>
              </div>
              <p className="text-lg font-semibold text-teal-400">{rec.followUp}</p>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="mt-6 text-xs text-slate-500 border-t border-slate-700 pt-4">
            These recommendations are based on algorithmic analysis and should be integrated with clinical judgment and patient-specific factors. Always follow institutional protocols and guidelines.
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}