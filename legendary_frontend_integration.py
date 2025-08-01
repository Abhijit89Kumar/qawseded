#!/usr/bin/env python3
"""
🔥 LEGENDARY Frontend Integration 🔥
- React frontend optimized for LEGENDARY CADENCE
- Real-time autocomplete with beam search
- Advanced product search interface
- Modern UI with animations
- Shock-worthy user experience
"""
import os
import json
from pathlib import Path
import structlog

logger = structlog.get_logger()

def create_legendary_package_json():
    """Create package.json for legendary frontend"""
    package_json = {
        "name": "legendary-cadence-frontend",
        "version": "1.0.0",
        "description": "🔥 LEGENDARY CADENCE Frontend - Shock-worthy E-commerce Search",
        "main": "src/index.js",
        "scripts": {
            "start": "react-scripts start",
            "build": "react-scripts build",
            "test": "react-scripts test",
            "eject": "react-scripts eject"
        },
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "react-scripts": "5.0.1",
            "axios": "^1.6.0",
            "styled-components": "^6.1.0",
            "@mui/material": "^5.14.0",
            "@mui/icons-material": "^5.14.0",
            "@emotion/react": "^11.11.0",
            "@emotion/styled": "^11.11.0",
            "framer-motion": "^10.16.0",
            "react-loading-skeleton": "^3.3.0",
            "lodash.debounce": "^4.0.8"
        },
        "eslintConfig": {
            "extends": [
                "react-app",
                "react-app/jest"
            ]
        },
        "browserslist": {
            "production": [
                ">0.2%",
                "not dead",
                "not op_mini all"
            ],
            "development": [
                "last 1 chrome version",
                "last 1 firefox version",
                "last 1 safari version"
            ]
        }
    }
    return package_json

def create_legendary_app_js():
    """Create the main App.js for legendary experience"""
    return '''import React, { useState, useCallback, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Container, Typography, Box, Chip } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import LegendarySearch from './components/LegendarySearch';
import LegendaryResults from './components/LegendaryResults';
import LegendaryStats from './components/LegendaryStats';
import './App.css';

// 🔥 LEGENDARY Theme
const legendaryTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#ff6b35', // Legendary orange
      light: '#ff9068',
      dark: '#c53d0b',
    },
    secondary: {
      main: '#4ecdc4', // Legendary teal
      light: '#7dede8',
      dark: '#259b95',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#cccccc',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 800,
      fontSize: '3.5rem',
      background: 'linear-gradient(45deg, #ff6b35 30%, #4ecdc4 90%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h2: {
      fontWeight: 700,
      fontSize: '2.5rem',
    },
  },
});

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Check backend health
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          setBackendStatus('connected');
          // Fetch stats
          const statsResponse = await fetch('http://localhost:8000/api/v1/stats');
          if (statsResponse.ok) {
            const statsData = await statsResponse.json();
            setStats(statsData);
          }
        } else {
          setBackendStatus('error');
        }
      } catch (error) {
        setBackendStatus('error');
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000); // Check every 10s
    return () => clearInterval(interval);
  }, []);

  const handleSearch = useCallback(async (query) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          max_results: 20,
          use_reranking: true,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.products);
      } else {
        console.error('Search failed');
        setSearchResults([]);
      }
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <ThemeProvider theme={legendaryTheme}>
      <CssBaseline />
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
      >
        <Container maxWidth="xl" sx={{ py: 4 }}>
          {/* Header */}
          <motion.div
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Box textAlign="center" mb={4}>
              <Typography variant="h1" component="h1" gutterBottom>
                🔥 LEGENDARY CADENCE
              </Typography>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                24.4M Parameter GRU-MN with Memory & Attention
              </Typography>
              
              {/* Status indicator */}
              <Box mt={2}>
                <Chip
                  label={`Backend: ${backendStatus === 'connected' ? '🟢 LIVE' : 
                                   backendStatus === 'error' ? '🔴 ERROR' : '🟡 CHECKING'}`}
                  color={backendStatus === 'connected' ? 'success' : 
                         backendStatus === 'error' ? 'error' : 'warning'}
                  variant="outlined"
                />
              </Box>
            </Box>
          </motion.div>

          {/* Stats */}
          {stats && (
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <LegendaryStats stats={stats} />
            </motion.div>
          )}

          {/* Search */}
          <motion.div
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <LegendarySearch
              onSearch={handleSearch}
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              isLoading={isLoading}
            />
          </motion.div>

          {/* Results */}
          <AnimatePresence>
            {(searchResults.length > 0 || isLoading) && (
              <motion.div
                initial={{ y: 50, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: -50, opacity: 0 }}
                transition={{ duration: 0.4 }}
              >
                <LegendaryResults
                  results={searchResults}
                  isLoading={isLoading}
                  searchQuery={searchQuery}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </Container>
      </motion.div>
    </ThemeProvider>
  );
}

export default App;'''

def create_legendary_search_component():
    """Create legendary search component with autocomplete"""
    return '''import React, { useState, useCallback, useEffect } from 'react';
import {
  Paper,
  TextField,
  Box,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Typography,
  Chip,
  CircularProgress
} from '@mui/material';
import { Search as SearchIcon, Clear as ClearIcon } from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { debounce } from 'lodash';

const LegendarySearch = ({ onSearch, searchQuery, setSearchQuery, isLoading }) => {
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);

  // Debounced autocomplete
  const debouncedAutocomplete = useCallback(
    debounce(async (query) => {
      if (!query.trim() || query.length < 2) {
        setSuggestions([]);
        setShowSuggestions(false);
        return;
      }

      setIsLoadingSuggestions(true);
      try {
        const response = await fetch('http://localhost:8000/api/v1/autocomplete', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query,
            max_suggestions: 8,
            beam_width: 5,
          }),
        });

        if (response.ok) {
          const data = await response.json();
          setSuggestions(data.suggestions);
          setShowSuggestions(data.suggestions.length > 0);
        }
      } catch (error) {
        console.error('Autocomplete error:', error);
        setSuggestions([]);
      } finally {
        setIsLoadingSuggestions(false);
      }
    }, 300),
    []
  );

  useEffect(() => {
    debouncedAutocomplete(searchQuery);
  }, [searchQuery, debouncedAutocomplete]);

  const handleInputChange = (event) => {
    setSearchQuery(event.target.value);
  };

  const handleSearch = () => {
    setShowSuggestions(false);
    onSearch(searchQuery);
  };

  const handleSuggestionClick = (suggestion) => {
    setSearchQuery(suggestion);
    setShowSuggestions(false);
    onSearch(suggestion);
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSearch();
    } else if (event.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  return (
    <Box sx={{ position: 'relative', mb: 4 }}>
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileFocus={{ scale: 1.02 }}
      >
        <Paper
          elevation={8}
          sx={{
            p: 2,
            background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
            border: '1px solid #ff6b35',
            borderRadius: 3,
          }}
        >
          <Box display="flex" alignItems="center">
            <TextField
              fullWidth
              variant="outlined"
              placeholder="🔥 Search with LEGENDARY CADENCE..."
              value={searchQuery}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
              sx={{
                '& .MuiOutlinedInput-root': {
                  fontSize: '1.2rem',
                  '& fieldset': {
                    border: 'none',
                  },
                },
                '& .MuiInputBase-input': {
                  color: 'white',
                },
                '& .MuiInputBase-input::placeholder': {
                  color: '#cccccc',
                },
              }}
            />
            
            {searchQuery && (
              <IconButton
                onClick={() => {
                  setSearchQuery('');
                  setSuggestions([]);
                  setShowSuggestions(false);
                }}
                sx={{ mx: 1 }}
              >
                <ClearIcon />
              </IconButton>
            )}
            
            <IconButton
              onClick={handleSearch}
              disabled={isLoading}
              sx={{
                bgcolor: 'primary.main',
                color: 'white',
                '&:hover': {
                  bgcolor: 'primary.dark',
                },
                '&:disabled': {
                  bgcolor: 'grey.600',
                },
              }}
            >
              {isLoading ? <CircularProgress size={24} /> : <SearchIcon />}
            </IconButton>
          </Box>
        </Paper>
      </motion.div>

      {/* Autocomplete suggestions */}
      <AnimatePresence>
        {showSuggestions && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            <Paper
              elevation={12}
              sx={{
                position: 'absolute',
                top: '100%',
                left: 0,
                right: 0,
                zIndex: 1000,
                mt: 1,
                maxHeight: 300,
                overflowY: 'auto',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                border: '1px solid #4ecdc4',
                borderRadius: 2,
              }}
            >
              {isLoadingSuggestions ? (
                <Box display="flex" justifyContent="center" p={2}>
                  <CircularProgress size={20} />
                  <Typography variant="body2" sx={{ ml: 1 }}>
                    Generating suggestions...
                  </Typography>
                </Box>
              ) : (
                <List dense>
                  {suggestions.map((suggestion, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <ListItem
                        button
                        onClick={() => handleSuggestionClick(suggestion)}
                        sx={{
                          '&:hover': {
                            bgcolor: 'rgba(255, 107, 53, 0.1)',
                            borderLeft: '3px solid #ff6b35',
                          },
                          transition: 'all 0.2s ease',
                        }}
                      >
                        <SearchIcon sx={{ mr: 1, color: 'secondary.main' }} />
                        <ListItemText
                          primary={suggestion}
                          sx={{
                            '& .MuiListItemText-primary': {
                              color: 'white',
                              fontWeight: 500,
                            },
                          }}
                        />
                        <Chip
                          label="🔥"
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      </ListItem>
                    </motion.div>
                  ))}
                </List>
              )}
            </Paper>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
};

export default LegendarySearch;'''

def create_legendary_results_component():
    """Create legendary results component"""
    return '''import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Rating,
  Skeleton,
  Paper
} from '@mui/material';
import { motion } from 'framer-motion';
import { AttachMoney, Star } from '@mui/icons-material';

const LegendaryResults = ({ results, isLoading, searchQuery }) => {
  if (isLoading) {
    return (
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          🔥 Searching with LEGENDARY AI...
        </Typography>
        <Grid container spacing={3}>
          {[...Array(6)].map((_, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card sx={{ bgcolor: 'background.paper' }}>
                <CardContent>
                  <Skeleton variant="text" width="80%" height={30} />
                  <Skeleton variant="text" width="60%" height={20} />
                  <Skeleton variant="text" width="40%" height={20} />
                  <Skeleton variant="rectangular" width="100%" height={60} />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (results.length === 0 && searchQuery) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Paper
          elevation={4}
          sx={{
            p: 4,
            textAlign: 'center',
            mt: 4,
            background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
            border: '1px solid #ff6b35',
          }}
        >
          <Typography variant="h6" gutterBottom>
            No results found for "{searchQuery}"
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try different keywords or check the spelling
          </Typography>
        </Paper>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          🔥 Found {results.length} LEGENDARY Results
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom mb={3}>
          Powered by 24.4M parameter GRU-MN with Memory & Attention
        </Typography>
        
        <Grid container spacing={3}>
          {results.map((product, index) => (
            <Grid item xs={12} sm={6} md={4} key={product.product_id}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                whileHover={{ scale: 1.03 }}
              >
                <Card
                  sx={{
                    height: '100%',
                    background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                    border: '1px solid #4ecdc4',
                    borderRadius: 2,
                    '&:hover': {
                      borderColor: '#ff6b35',
                      boxShadow: '0 8px 25px rgba(255, 107, 53, 0.3)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  <CardContent>
                    {/* Product Title */}
                    <Typography
                      variant="h6"
                      component="h3"
                      gutterBottom
                      sx={{
                        fontWeight: 600,
                        color: 'white',
                        lineHeight: 1.3,
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}
                    >
                      {product.title}
                    </Typography>

                    {/* Category */}
                    <Chip
                      label={product.category}
                      size="small"
                      color="secondary"
                      variant="outlined"
                      sx={{ mb: 2 }}
                    />

                    {/* Description */}
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        mb: 2,
                        display: '-webkit-box',
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}
                    >
                      {product.description}
                    </Typography>

                    {/* Price and Rating */}
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      {product.price && (
                        <Box display="flex" alignItems="center">
                          <AttachMoney sx={{ color: 'primary.main' }} />
                          <Typography variant="h6" color="primary.main" fontWeight="bold">
                            {product.price.toFixed(2)}
                          </Typography>
                        </Box>
                      )}
                      
                      {product.rating && (
                        <Box display="flex" alignItems="center">
                          <Rating
                            value={product.rating}
                            readOnly
                            size="small"
                            sx={{
                              '& .MuiRating-iconFilled': {
                                color: '#ffd700',
                              },
                            }}
                          />
                          <Typography variant="body2" sx={{ ml: 0.5 }}>
                            ({product.rating.toFixed(1)})
                          </Typography>
                        </Box>
                      )}
                    </Box>

                    {/* Relevance Score */}
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2" color="text.secondary">
                        Relevance: {(product.relevance_score * 100).toFixed(1)}%
                      </Typography>
                      <Chip
                        label="🔥 LEGENDARY"
                        size="small"
                        color="primary"
                        variant="filled"
                      />
                    </Box>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </Box>
    </motion.div>
  );
};

export default LegendaryResults;'''

def create_legendary_stats_component():
    """Create legendary stats component"""
    return '''import React from 'react';
import { Paper, Grid, Typography, Box, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import {
  Memory,
  Psychology,
  Dataset,
  Speed,
  Category,
  Search
} from '@mui/icons-material';

const LegendaryStats = ({ stats }) => {
  const statItems = [
    {
      icon: <Memory />,
      label: 'Parameters',
      value: stats.model_info.total_parameters,
      color: '#ff6b35'
    },
    {
      icon: <Psychology />,
      label: 'Architecture',
      value: 'GRU-MN + Memory',
      color: '#4ecdc4'
    },
    {
      icon: <Dataset />,
      label: 'Vocabulary',
      value: `${stats.data_stats.vocabulary_size:,}`,
      color: '#ff6b35'
    },
    {
      icon: <Search />,
      label: 'Queries',
      value: `${stats.data_stats.total_queries:,}`,
      color: '#4ecdc4'
    },
    {
      icon: <Category />,
      label: 'Categories',
      value: `${stats.data_stats.query_categories + stats.data_stats.product_categories}`,
      color: '#ff6b35'
    },
    {
      icon: <Speed />,
      label: 'Status',
      value: stats.performance.status,
      color: '#4ecdc4'
    }
  ];

  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Paper
        elevation={6}
        sx={{
          p: 3,
          mb: 4,
          background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
          border: '1px solid #4ecdc4',
          borderRadius: 3,
        }}
      >
        <Typography variant="h6" gutterBottom sx={{ textAlign: 'center', mb: 3 }}>
          🔥 LEGENDARY Model Statistics
        </Typography>
        
        <Grid container spacing={2}>
          {statItems.map((item, index) => (
            <Grid item xs={6} sm={4} md={2} key={index}>
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
              >
                <Box
                  textAlign="center"
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    background: `linear-gradient(135deg, ${item.color}20 0%, ${item.color}10 100%)`,
                    border: `1px solid ${item.color}40`,
                    '&:hover': {
                      background: `linear-gradient(135deg, ${item.color}30 0%, ${item.color}15 100%)`,
                      borderColor: item.color,
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  <Box
                    sx={{
                      color: item.color,
                      mb: 1,
                      display: 'flex',
                      justifyContent: 'center',
                    }}
                  >
                    {item.icon}
                  </Box>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ fontSize: '0.75rem' }}
                  >
                    {item.label}
                  </Typography>
                  <Typography
                    variant="body1"
                    fontWeight="bold"
                    sx={{ color: 'white', fontSize: '0.9rem' }}
                  >
                    {item.value}
                  </Typography>
                </Box>
              </motion.div>
            </Grid>
          ))}
        </Grid>
        
        <Box textAlign="center" mt={3}>
          <Chip
            label={`Device: ${stats.performance.device.toUpperCase()}`}
            color="primary"
            variant="outlined"
            sx={{ mr: 1 }}
          />
          <Chip
            label="SHOCK-WORTHY PERFORMANCE"
            color="secondary"
            variant="filled"
          />
        </Box>
      </Paper>
    </motion.div>
  );
};

export default LegendaryStats;'''

def create_legendary_css():
    """Create legendary CSS with animations"""
    return '''/* 🔥 LEGENDARY CADENCE CSS */

body {
  margin: 0;
  font-family: 'Roboto', 'Helvetica', 'Arial', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
  min-height: 100vh;
}

* {
  box-sizing: border-box;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
  background: linear-gradient(135deg, #ff6b35 0%, #4ecdc4 100%);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(135deg, #e55a2b 0%, #3bb5ae 100%);
}

/* Legendary animations */
@keyframes legendaryPulse {
  0% {
    box-shadow: 0 0 0 0 rgba(255, 107, 53, 0.7);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(255, 107, 53, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(255, 107, 53, 0);
  }
}

@keyframes legendaryGlow {
  from {
    text-shadow: 0 0 5px #ff6b35, 0 0 10px #ff6b35, 0 0 15px #ff6b35;
  }
  to {
    text-shadow: 0 0 10px #4ecdc4, 0 0 20px #4ecdc4, 0 0 30px #4ecdc4;
  }
}

.legendary-pulse {
  animation: legendaryPulse 2s infinite;
}

.legendary-glow {
  animation: legendaryGlow 2s ease-in-out infinite alternate;
}

/* Legendary gradient text */
.legendary-gradient-text {
  background: linear-gradient(45deg, #ff6b35 30%, #4ecdc4 90%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Loading animations */
@keyframes legendarySpinner {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.legendary-spinner {
  animation: legendarySpinner 1s linear infinite;
}

/* Card hover effects */
.legendary-card {
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.legendary-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 107, 53, 0.1),
    transparent
  );
  transition: all 0.5s;
}

.legendary-card:hover::before {
  left: 100%;
}

/* Legendary button styles */
.legendary-button {
  background: linear-gradient(45deg, #ff6b35 30%, #4ecdc4 90%);
  border: none;
  border-radius: 25px;
  color: white;
  padding: 12px 24px;
  font-weight: bold;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.legendary-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(255, 107, 53, 0.4);
}

.legendary-button:active {
  transform: translateY(0);
}

/* Responsive design */
@media (max-width: 768px) {
  .legendary-title {
    font-size: 2.5rem !important;
  }
  
  .legendary-subtitle {
    font-size: 1rem !important;
  }
}

@media (max-width: 480px) {
  .legendary-title {
    font-size: 2rem !important;
  }
}

/* Print styles */
@media print {
  body {
    background: white !important;
    color: black !important;
  }
  
  .legendary-gradient-text {
    -webkit-text-fill-color: black !important;
    background: none !important;
  }
}'''

def setup_legendary_frontend():
    """Setup the complete legendary frontend"""
    logger.info("🔥 Setting up LEGENDARY Frontend...")
    
    # Create frontend directory
    frontend_dir = Path("legendary_frontend")
    frontend_dir.mkdir(exist_ok=True)
    
    # Create src directory
    src_dir = frontend_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    # Create components directory
    components_dir = src_dir / "components"
    components_dir.mkdir(exist_ok=True)
    
    # Create public directory
    public_dir = frontend_dir / "public"
    public_dir.mkdir(exist_ok=True)
    
    logger.info("   Creating package.json...")
    with open(frontend_dir / "package.json", 'w', encoding='utf-8') as f:
        json.dump(create_legendary_package_json(), f, indent=2)
    
    logger.info("   Creating App.js...")
    with open(src_dir / "App.js", 'w', encoding='utf-8') as f:
        f.write(create_legendary_app_js())
    
    logger.info("   Creating components...")
    with open(components_dir / "LegendarySearch.js", 'w', encoding='utf-8') as f:
        f.write(create_legendary_search_component())
    
    with open(components_dir / "LegendaryResults.js", 'w', encoding='utf-8') as f:
        f.write(create_legendary_results_component())
    
    with open(components_dir / "LegendaryStats.js", 'w', encoding='utf-8') as f:
        f.write(create_legendary_stats_component())
    
    logger.info("   Creating CSS...")
    with open(src_dir / "App.css", 'w', encoding='utf-8') as f:
        f.write(create_legendary_css())
    
    # Create index.js
    logger.info("   Creating index.js...")
    index_js = '''import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);'''
    
    with open(src_dir / "index.js", 'w', encoding='utf-8') as f:
        f.write(index_js)
    
    # Create index.html
    logger.info("   Creating index.html...")
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#ff6b35" />
  <meta name="description" content="🔥 LEGENDARY CADENCE - 24.4M Parameter E-commerce Search" />
  <title>🔥 LEGENDARY CADENCE</title>
  <style>
    body {
      margin: 0;
      background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }
  </style>
</head>
<body>
  <noscript>You need to enable JavaScript to run this app.</noscript>
  <div id="root"></div>
</body>
</html>'''
    
    with open(public_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    logger.info("✅ LEGENDARY Frontend setup complete!")
    return frontend_dir

def main():
    """Main function"""
    logger.info("🔥🔥🔥 LEGENDARY FRONTEND INTEGRATION 🔥🔥🔥")
    
    try:
        frontend_dir = setup_legendary_frontend()
        
        logger.info("")
        logger.info("🔥 LEGENDARY Frontend created!")
        logger.info("Features:")
        logger.info("• React 18 with Material-UI")
        logger.info("• Framer Motion animations")
        logger.info("• Real-time autocomplete with beam search")
        logger.info("• Advanced product search interface")
        logger.info("• Responsive design")
        logger.info("• SHOCK-WORTHY UI/UX")
        logger.info("")
        logger.info("🚀 To start the frontend:")
        logger.info(f"   cd {frontend_dir}")
        logger.info("   npm install")
        logger.info("   npm start")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Frontend setup failed: {e}")
        return False

if __name__ == "__main__":
    main()