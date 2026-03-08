import React from 'react';
import { Link } from 'react-router-dom';
import { createPageUrl } from '@/utils';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Stethoscope, 
  Baby, 
  Activity, 
  Brain, 
  Shield, 
  Sparkles,
  ArrowRight,
  CheckCircle2,
  BarChart3,
  Lightbulb
} from 'lucide-react';
import { motion } from 'framer-motion';
import SystemArchitecture from '@/components/maternal/SystemArchitecture';

export default function Home() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-8 lg:py-12"
      >
        <motion.div
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-gradient-to-br from-teal-400 to-cyan-500 shadow-2xl shadow-teal-500/30 mb-6"
        >
          <Activity className="w-10 h-10 text-white" />
        </motion.div>
        <h1 className="text-3xl lg:text-5xl font-bold text-white mb-4">
          Hybrid Clinical–AI
          <span className="block text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400">
            Maternal Risk Detection
          </span>
        </h1>
        <p className="text-slate-400 max-w-2xl mx-auto text-lg">
          Advanced machine learning ensemble combined with clinical expertise for accurate pregnancy risk assessment
        </p>
        <div className="flex flex-wrap justify-center gap-3 mt-6">
          <Badge className="bg-teal-500/20 text-teal-400 border border-teal-500/30 px-4 py-1">
            <Brain className="w-3 h-3 mr-1" /> Ensemble ML
          </Badge>
          <Badge className="bg-violet-500/20 text-violet-400 border border-violet-500/30 px-4 py-1">
            <Lightbulb className="w-3 h-3 mr-1" /> Explainable AI
          </Badge>
          <Badge className="bg-amber-500/20 text-amber-400 border border-amber-500/30 px-4 py-1">
            <BarChart3 className="w-3 h-3 mr-1" /> 87.19% Accuracy
          </Badge>
        </div>
      </motion.div>

      {/* Navigation Cards */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Doctor Interface Card */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Link to={createPageUrl('DoctorMode')}>
            <Card className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl border border-white/10 hover:border-teal-500/50 transition-all duration-300 cursor-pointer group overflow-hidden">
              <CardContent className="p-8">
                <div className="flex items-start gap-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-teal-500/20 group-hover:scale-110 transition-transform">
                    <Stethoscope className="w-8 h-8 text-white" />
                  </div>
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-white mb-2 group-hover:text-teal-400 transition-colors">
                      Doctor Interface
                    </h2>
                    <p className="text-slate-400 mb-4">
                      Advanced analytics dashboard with model comparisons, explainability visualizations, and comprehensive evaluation metrics.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {['Model Comparison', 'SHAP Analysis', 'Feature Selection', 'Evaluation'].map((item) => (
                        <Badge key={item} variant="outline" className="bg-slate-700/50 text-slate-300 border-slate-600">
                          <CheckCircle2 className="w-3 h-3 mr-1 text-teal-400" />
                          {item}
                        </Badge>
                      ))}
                    </div>
                    <Button className="mt-6 bg-teal-500 hover:bg-teal-400 text-white group-hover:translate-x-2 transition-transform">
                      Open Dashboard <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        </motion.div>

        {/* Patient Interface Card */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Link to={createPageUrl('PatientMode')}>
            <Card className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl border border-white/10 hover:border-pink-500/50 transition-all duration-300 cursor-pointer group overflow-hidden">
              <CardContent className="p-8">
                <div className="flex items-start gap-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-pink-400 to-rose-500 flex items-center justify-center shadow-lg shadow-pink-500/20 group-hover:scale-110 transition-transform">
                    <Baby className="w-8 h-8 text-white" />
                  </div>
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-white mb-2 group-hover:text-pink-400 transition-colors">
                      Patient Interface
                    </h2>
                    <p className="text-slate-400 mb-4">
                      Simple, calming interface for health screening with easy-to-understand results and personalized recommendations.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {['Simple Input', 'Clear Results', 'Recommendations', 'Privacy-Safe'].map((item) => (
                        <Badge key={item} variant="outline" className="bg-slate-700/50 text-slate-300 border-slate-600">
                          <CheckCircle2 className="w-3 h-3 mr-1 text-pink-400" />
                          {item}
                        </Badge>
                      ))}
                    </div>
                    <Button className="mt-6 bg-pink-500 hover:bg-pink-400 text-white group-hover:translate-x-2 transition-transform">
                      Start Screening <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        </motion.div>
      </div>

      {/* About Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Card className="bg-gradient-to-br from-teal-500/10 to-cyan-500/10 backdrop-blur-xl border border-teal-500/20">
          <CardContent className="p-6 lg:p-8">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-xl bg-teal-500/20 flex items-center justify-center flex-shrink-0">
                <Shield className="w-6 h-6 text-teal-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white mb-2">Hybrid ML + Clinical Framework</h3>
                <p className="text-slate-300 leading-relaxed">
                  This system combines multiple machine learning models (XGBoost, Random Forest, Gradient Boosting) 
                  with clinical guidelines to provide accurate pregnancy risk predictions. The ensemble approach 
                  achieves <span className="text-teal-400 font-semibold">87.19% accuracy</span> while maintaining 
                  full explainability through SHAP analysis and feature importance visualization.
                </p>
                <div className="mt-4 flex items-center gap-2 text-sm text-amber-400">
                  <Sparkles className="w-4 h-4" />
                  <span>This is an AI-assisted screening tool — not a diagnostic replacement.</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* System Architecture */}
      <SystemArchitecture />
    </div>
  );
}