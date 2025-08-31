'use client';

import { useState, useEffect } from 'react';
import { Lightbulb, CheckCircle, XCircle, MessageSquare, Clock, Zap } from 'lucide-react';

interface Suggestion {
  id: string;
  timestamp: string;
  type: string;
  content: string;
  rationale: string;
  confidence: number;
  priority: number;
  actionable: boolean;
  estimated_value: number;
  context: Record<string, any>;
}

interface SuggestionsResponse {
  suggestions: Suggestion[];
  count: number;
  timestamp: string;
}

interface SuggestionsPanelProps {
  isVisible: boolean;
  onClose: () => void;
  onSuggestionSelect: (suggestion: Suggestion) => void;
}

export default function SuggestionsPanel({ isVisible, onClose, onSuggestionSelect }: SuggestionsPanelProps) {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState<string | null>(null);

  useEffect(() => {
    if (isVisible) {
      fetchSuggestions();
    }
  }, [isVisible]);

  const fetchSuggestions = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/suggestions');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch suggestions: ${response.statusText}`);
      }
      
      const data: SuggestionsResponse = await response.json();
      setSuggestions(data.suggestions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch suggestions');
    } finally {
      setIsLoading(false);
    }
  };

  const provideFeedback = async (suggestionId: string, accepted: boolean, feedback?: string) => {
    setFeedbackSubmitting(suggestionId);
    
    try {
      const response = await fetch('http://localhost:8000/api/suggestions/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          suggestion_id: suggestionId,
          accepted,
          feedback: feedback || null
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }
      
      // Remove the suggestion from the list after feedback
      setSuggestions(prev => prev.filter(s => s.id !== suggestionId));
      
    } catch (err) {
      console.error('Error submitting feedback:', err);
    } finally {
      setFeedbackSubmitting(null);
    }
  };

  const handleAccept = (suggestion: Suggestion) => {
    provideFeedback(suggestion.id, true, 'User accepted suggestion');
    onSuggestionSelect(suggestion);
  };

  const handleReject = (suggestion: Suggestion) => {
    provideFeedback(suggestion.id, false, 'User rejected suggestion');
  };

  const getSuggestionIcon = (type: string) => {
    switch (type) {
      case 'action':
        return <Zap className="w-4 h-4 text-blue-500" />;
      case 'question':
        return <MessageSquare className="w-4 h-4 text-green-500" />;
      case 'insight':
        return <Lightbulb className="w-4 h-4 text-yellow-500" />;
      case 'reminder':
        return <Clock className="w-4 h-4 text-purple-500" />;
      case 'exploration':
        return <Lightbulb className="w-4 h-4 text-orange-500" />;
      default:
        return <Lightbulb className="w-4 h-4 text-gray-500" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-gray-100 text-gray-800';
  };

  const getPriorityColor = (priority: number) => {
    if (priority >= 0.7) return 'bg-red-100 text-red-800';
    if (priority >= 0.4) return 'bg-orange-100 text-orange-800';
    return 'bg-blue-100 text-blue-800';
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            <Lightbulb className="w-6 h-6 text-yellow-500" />
            <h2 className="text-xl font-semibold text-gray-900">Proactive Suggestions</h2>
            {suggestions.length > 0 && (
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm">
                {suggestions.length} suggestion{suggestions.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <XCircle className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              <span className="ml-3 text-gray-600">Loading suggestions...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <XCircle className="w-12 h-12 text-red-500 mx-auto mb-3" />
                <p className="text-red-600 font-medium">Error loading suggestions</p>
                <p className="text-gray-500 text-sm mt-1">{error}</p>
                <button
                  onClick={fetchSuggestions}
                  className="mt-3 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : suggestions.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <Lightbulb className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-500 font-medium">No suggestions available</p>
                <p className="text-gray-400 text-sm mt-1">The AI is still analyzing patterns. Check back later!</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {suggestions.map((suggestion) => (
                <div key={suggestion.id} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      {/* Header */}
                      <div className="flex items-center gap-2 mb-2">
                        {getSuggestionIcon(suggestion.type)}
                        <span className="text-sm font-medium text-gray-600 capitalize">
                          {suggestion.type}
                        </span>
                        <span className={`text-xs px-2 py-1 rounded-full ${getConfidenceColor(suggestion.confidence)}`}>
                          {Math.round(suggestion.confidence * 100)}% confident
                        </span>
                        <span className={`text-xs px-2 py-1 rounded-full ${getPriorityColor(suggestion.priority)}`}>
                          {suggestion.priority >= 0.7 ? 'High' : suggestion.priority >= 0.4 ? 'Medium' : 'Low'} priority
                        </span>
                      </div>

                      {/* Content */}
                      <p className="text-gray-800 mb-2 leading-relaxed">
                        {suggestion.content}
                      </p>

                      {/* Rationale */}
                      <p className="text-sm text-gray-600 italic mb-3">
                        {suggestion.rationale}
                      </p>

                      {/* Context */}
                      {suggestion.context && Object.keys(suggestion.context).length > 0 && (
                        <div className="text-xs text-gray-500 mb-3">
                          <strong>Context:</strong> {JSON.stringify(suggestion.context, null, 2)}
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleAccept(suggestion)}
                          disabled={feedbackSubmitting === suggestion.id}
                          className="flex items-center gap-1 px-3 py-1.5 bg-green-500 text-white text-sm rounded-lg hover:bg-green-600 transition-colors disabled:opacity-50"
                        >
                          <CheckCircle className="w-3 h-3" />
                          Accept
                        </button>
                        <button
                          onClick={() => handleReject(suggestion)}
                          disabled={feedbackSubmitting === suggestion.id}
                          className="flex items-center gap-1 px-3 py-1.5 bg-red-500 text-white text-sm rounded-lg hover:bg-red-600 transition-colors disabled:opacity-50"
                        >
                          <XCircle className="w-3 h-3" />
                          Dismiss
                        </button>
                      </div>
                    </div>

                    {/* Value indicator */}
                    <div className="ml-4 text-right">
                      <div className="text-xs text-gray-500 mb-1">Estimated Value</div>
                      <div className="w-16 h-2 bg-gray-200 rounded-full">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${suggestion.estimated_value * 100}%` }}
                        />
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {Math.round(suggestion.estimated_value * 100)}%
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t bg-gray-50">
          <p className="text-sm text-gray-600">
            Suggestions are generated from memory patterns and reflection insights
          </p>
          <button
            onClick={fetchSuggestions}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
} 