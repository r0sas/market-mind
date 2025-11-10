"""Spanish translations"""

TRANSLATIONS_ES = {
    # Page titles
    'page_title': 'Calculadora de Valor Intrínseco',
    'page_subtitle': 'Análisis profesional de valoración de acciones con IA',
    
    # Header
    'enter_tickers': 'Ingrese Símbolos de Acciones',
    'ticker_placeholder': 'ej., AAPL, MSFT, GOOGL',
    'ticker_help': 'Ingrese uno o más símbolos separados por comas',
    'calculate': 'Calcular Valor Intrínseco',
    'enter_ticker_error': 'Por favor ingrese al menos un símbolo',
    
    # Sidebar
    'sidebar_title': '⚙️ Configuración',
    'model_selection_mode': 'Modo de Selección de Modelos',
    'model_selection_question': '¿Cómo desea seleccionar los modelos?',
    'auto_select': '✨ Selección Automática (Recomendado)',
    'manual_select': '📋 Selección Manual',
    'auto_select_help': 'La IA analiza los datos y selecciona los mejores modelos',
    'smart_enabled': '✨ Selección inteligente habilitada',
    'min_fit_score': 'Puntuación Mínima de Ajuste',
    'min_fit_score_help': 'Puntuaciones más altas = mejor ajuste del modelo',
    'show_excluded': 'Mostrar Modelos Excluidos',
    'show_excluded_help': 'Mostrar modelos que no cumplieron el umbral',
    'manual_active': '📋 Selección manual activa',
    'model_selection': 'Seleccionar Modelos de Valoración',
    'select_models': 'Elija los modelos a usar',
    'select_models_help': 'Seleccione uno o más modelos',
    
    # Parameters
    'advanced_params': 'Parámetros Avanzados',
    'dcf_params': 'Parámetros DCF',
    'discount_rate': 'Tasa de Descuento (%)',
    'discount_rate_help': 'Tasa de retorno requerida (WACC)',
    'terminal_growth': 'Tasa de Crecimiento Terminal (%)',
    'terminal_growth_help': 'Tasa de crecimiento a largo plazo',
    
    # Analysis options
    'analysis_options': 'Opciones de Análisis',
    'show_confidence': 'Mostrar Puntuaciones de Confianza',
    'show_warnings': 'Mostrar Advertencias de Datos',
    'weighted_avg': 'Usar Promedio Ponderado',
    'weighted_avg_help': 'Ponderar modelos por confianza',
    'margin_safety': 'Margen de Seguridad Objetivo (%)',
    'margin_safety_help': 'Colchón de seguridad bajo valor intrínseco',
    
    # AI
    'ai_insights': '🤖 Análisis IA',
    'enable_ai': 'Habilitar Análisis IA',
    'enable_ai_help': 'Obtener análisis y recomendaciones con IA',
    'groq_key': 'Clave API Groq',
    'groq_key_help': 'Obtenga clave gratis en console.groq.com',
    'api_provided': '✅ Clave API proporcionada',
    'api_info': 'Obtenga una clave gratis en console.groq.com',
    'ai_init_failed': 'Falló la inicialización de IA',
    
    # Results
    'processing': 'Procesando',
    'fetching': 'Obteniendo datos de',
    'success': 'Analizadas exitosamente',
    'stocks': 'acciones',
    'data_warnings': '⚠️ Advertencias de Calidad de Datos',
    'warnings': 'advertencias',
    
    # Tables
    'current_price': 'Precio Actual',
    'models_selected': 'Modelos Seleccionados',
    'iv_summary': '📊 Resumen de Valor Intrínseco',
    'iv_comparison': 'Comparación de Valor Intrínseco',
    'confidence_scores': '🎯 Puntuaciones de Confianza',
    
    # Model analysis
    'model_analysis': '🔍 Análisis de Selección de Modelos',
    'model_analysis_subtitle': '*Análisis de ajuste del modelo con IA*',
    'model_details': 'Detalles del Modelo',
    'highly_recommended': '**Altamente Recomendado** (0.70+)',
    'recommended': '**Recomendado** (0.50-0.69)',
    'marginal': '**Marginal** (0.30-0.49)',
    'not_suitable': '**No Adecuado** (<0.30)',
    'score_legend': '**Leyenda de Puntuación:**',
    'avg_fit_score': 'Puntuación de Ajuste Promedio',
    'detailed_analysis': 'Análisis Detallado',
    'selected_models': '✅ Modelos Seleccionados',
    'score': 'Puntuación',
    'strengths': '**Fortalezas:**',
    'considerations': '**Consideraciones:**',
    'excluded_models': '❌ Modelos Excluidos',
    'primary_reason': '**Razón Principal:**',
    'issues': '**Problemas:**',
    'positive_factors': '**Factores Positivos:**',
    'target': 'Objetivo',
    
    # Margin of safety
    'margin_analysis': '💰 Análisis de Margen de Seguridad',
    'margin_by_model': 'Margen de Seguridad por Modelo',
    
    # Sensitivity
    'sensitivity': 'Análisis de Sensibilidad',
    'enable_sensitivity': 'Habilitar Análisis de Sensibilidad',
    'model_analyze': 'Modelo a Analizar',
    'param_vary': 'Parámetro a Variar',
    'sensitivity_analysis': 'Análisis de Sensibilidad',
    'sensitivity_info': 'Este gráfico muestra cómo cambia el valor intrínseco de {ticker} al variar {param}',
    'sensitivity_failed': 'Falló el análisis de sensibilidad',
    'sensitivity_single_only': '⚠️ Análisis de sensibilidad solo disponible para un ticker',
    
    # Export
    'export': '📥 Exportar Resultados',
    'download_valuations': '📄 Descargar Valoraciones (CSV)',
    'download_margins': '📄 Descargar Análisis de Margen (CSV)',
    'download_report': '📄 Descargar Reporte Completo (TXT)',
    
    # AI Analysis
    'ai_analysis': '🤖 Análisis con IA',
    'ai_subtitle': '*Perspectivas generadas basadas en modelos de valoración*',
    'ai_caption': '💡 Análisis generado por IA. Siempre haga su propia investigación.',
    
    # Errors
    'no_analysis': '❌ No se analizaron acciones exitosamente',
    'failed_analyze': '⚠️ No se pudo analizar',
    'troubleshooting': '**Solución de problemas:**',
    'verify_ticker': '• Verifique que los símbolos sean correctos',
    'check_history': '• Asegúrese de que las acciones tengan datos históricos (5+ años)',
    'reit_warning': '• Los REITs pueden no funcionar con todos los modelos',
    'try_later': '• Intente más tarde si la fuente de datos no está disponible',
    
    # Info sections
    'about_models': '📚 Acerca de los Modelos de Valoración',
    'faq': '❓ Preguntas Frecuentes',
    'technical': '⚙️ Detalles Técnicos',
    'tip': '💡 **Consejo:** ¡Comience con el modo automático para ver qué modelos funcionan mejor!',
    
    # Disclaimer
    'disclaimer': '⚠️ Este análisis es solo informativo. No es asesoría financiera. Siempre consulte a un asesor calificado.'
}