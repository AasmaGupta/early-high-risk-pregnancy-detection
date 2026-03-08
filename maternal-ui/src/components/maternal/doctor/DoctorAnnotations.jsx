import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { 
  MessageSquare, 
  Tag, 
  AlertTriangle,
  Save,
  Plus,
  X,
  Clock,
  User,
  Edit3
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const predefinedTags = [
  { label: 'High Priority', color: 'rose' },
  { label: 'Follow-up Required', color: 'amber' },
  { label: 'Reviewed', color: 'emerald' },
  { label: 'Specialist Referral', color: 'purple' },
  { label: 'Patient Education', color: 'blue' },
  { label: 'Monitoring', color: 'cyan' },
];

const tagColors = {
  rose: 'bg-rose-500/20 text-rose-400 border-rose-500/30',
  amber: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  emerald: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  purple: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  blue: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  cyan: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
};

export default function DoctorAnnotations({ predictionResult }) {
  const [notes, setNotes] = useState('');
  const [selectedTags, setSelectedTags] = useState([]);
  const [overrideRisk, setOverrideRisk] = useState(null);
  const [savedAnnotations, setSavedAnnotations] = useState([]);
  const [isEditing, setIsEditing] = useState(false);

  const toggleTag = (tag) => {
    if (selectedTags.includes(tag.label)) {
      setSelectedTags(selectedTags.filter(t => t !== tag.label));
    } else {
      setSelectedTags([...selectedTags, tag.label]);
    }
  };

  const handleSave = () => {
    if (!notes.trim() && selectedTags.length === 0 && !overrideRisk) return;

    const newAnnotation = {
      id: Date.now(),
      notes,
      tags: selectedTags,
      overrideRisk,
      originalRisk: predictionResult?.riskLevel,
      timestamp: new Date().toISOString(),
      author: 'Dr. Smith', // Would come from auth in real app
    };

    setSavedAnnotations([newAnnotation, ...savedAnnotations]);
    setNotes('');
    setSelectedTags([]);
    setOverrideRisk(null);
    setIsEditing(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
              <MessageSquare className="w-5 h-5 text-white" />
            </div>
            Clinical Annotations
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {/* New Annotation Form */}
          <div className="space-y-4 mb-6">
            {/* Clinical Notes */}
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">Clinical Notes</label>
              <Textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add clinical observations, patient history notes, or decision rationale..."
                className="bg-slate-700/50 border-slate-600 text-white placeholder:text-slate-500 min-h-[100px]"
              />
            </div>

            {/* Tags */}
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block flex items-center gap-2">
                <Tag className="w-4 h-4" />
                Case Tags
              </label>
              <div className="flex flex-wrap gap-2">
                {predefinedTags.map((tag) => (
                  <button
                    key={tag.label}
                    onClick={() => toggleTag(tag)}
                    className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${
                      selectedTags.includes(tag.label)
                        ? tagColors[tag.color]
                        : 'bg-slate-700/50 text-slate-400 border-slate-600 hover:border-slate-500'
                    }`}
                  >
                    {selectedTags.includes(tag.label) && '✓ '}
                    {tag.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Risk Override */}
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                Clinical Risk Override
                {predictionResult && (
                  <span className="text-xs text-slate-500 ml-2">
                    (Model prediction: {predictionResult.riskLevel})
                  </span>
                )}
              </label>
              <div className="flex gap-2">
                {['low', 'mid', 'high'].map((level) => (
                  <button
                    key={level}
                    onClick={() => setOverrideRisk(overrideRisk === level ? null : level)}
                    className={`flex-1 py-2 rounded-lg font-medium transition-all ${
                      overrideRisk === level
                        ? level === 'low' ? 'bg-emerald-500 text-white' :
                          level === 'mid' ? 'bg-amber-500 text-white' :
                          'bg-rose-500 text-white'
                        : 'bg-slate-700/50 text-slate-400 border border-slate-600 hover:border-slate-500'
                    }`}
                  >
                    {level.charAt(0).toUpperCase() + level.slice(1)}
                  </button>
                ))}
              </div>
              {overrideRisk && (
                <p className="text-xs text-amber-400 mt-2">
                  ⚠️ You are overriding the model prediction. This will be documented.
                </p>
              )}
            </div>

            {/* Save Button */}
            <Button
              onClick={handleSave}
              disabled={!notes.trim() && selectedTags.length === 0 && !overrideRisk}
              className="w-full bg-teal-500 hover:bg-teal-400 text-white"
            >
              <Save className="w-4 h-4 mr-2" />
              Save Annotation
            </Button>
          </div>

          {/* Saved Annotations */}
          {savedAnnotations.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-slate-300 mb-3">Previous Annotations</h4>
              <div className="space-y-3">
                <AnimatePresence>
                  {savedAnnotations.map((annotation) => (
                    <motion.div
                      key={annotation.id}
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                          <User className="w-3 h-3" />
                          {annotation.author}
                          <Clock className="w-3 h-3 ml-2" />
                          {new Date(annotation.timestamp).toLocaleString()}
                        </div>
                        {annotation.overrideRisk && (
                          <Badge className="bg-amber-500/20 text-amber-400 border border-amber-500/30 text-xs">
                            Override: {annotation.overrideRisk}
                          </Badge>
                        )}
                      </div>
                      
                      {annotation.notes && (
                        <p className="text-sm text-slate-300 mb-2">{annotation.notes}</p>
                      )}
                      
                      {annotation.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {annotation.tags.map((tagLabel) => {
                            const tag = predefinedTags.find(t => t.label === tagLabel);
                            return (
                              <span
                                key={tagLabel}
                                className={`px-2 py-0.5 rounded-full text-xs border ${tagColors[tag?.color || 'cyan']}`}
                              >
                                {tagLabel}
                              </span>
                            );
                          })}
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          )}

          {savedAnnotations.length === 0 && (
            <div className="text-center py-6 text-slate-500">
              <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No annotations yet</p>
              <p className="text-xs">Add clinical notes or case tags above</p>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}