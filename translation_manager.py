"""
Translation Manager UI
A Streamlit app to manage and edit translations
"""

import streamlit as st
import json
import pandas as pd
from translations import TRANSLATIONS, get_available_languages
from typing import Dict

st.set_page_config(
    page_title="Translation Manager",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Translation Manager")
st.markdown("Manage translations for the Intrinsic Value Calculator")

# Sidebar controls
st.sidebar.title("Controls")

# Select language to edit
languages = get_available_languages()
selected_lang = st.sidebar.selectbox(
    "Select Language to Edit",
    options=list(languages.keys()),
    format_func=lambda x: languages[x]
)

# Compare with another language
compare_with = st.sidebar.selectbox(
    "Compare With",
    options=['None'] + list(languages.keys()),
    index=0
)

# Filter options
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

search_term = st.sidebar.text_input("🔍 Search Keys", "")
show_only_missing = st.sidebar.checkbox("Show Only Missing Translations", False)
show_only_untranslated = st.sidebar.checkbox("Show Only Untranslated (same as English)", False)

# Stats
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistics")

en_keys = set(TRANSLATIONS['en'].keys())
lang_keys = set(TRANSLATIONS[selected_lang].keys())

total_keys = len(en_keys)
translated_keys = len(lang_keys)
missing_keys = len(en_keys - lang_keys)
completion_pct = (translated_keys / total_keys * 100) if total_keys > 0 else 0

st.sidebar.metric("Total Keys", total_keys)
st.sidebar.metric("Translated", translated_keys)
st.sidebar.metric("Missing", missing_keys)
st.sidebar.progress(completion_pct / 100)
st.sidebar.caption(f"Completion: {completion_pct:.1f}%")

# Export/Import
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Export/Import")

if st.sidebar.button("📥 Export Current Language (JSON)"):
    json_data = json.dumps(TRANSLATIONS[selected_lang], indent=2, ensure_ascii=False)
    st.sidebar.download_button(
        label="Download JSON",
        data=json_data,
        file_name=f"translations_{selected_lang}.json",
        mime="application/json"
    )

# Main content
tabs = st.tabs(["📝 Edit Translations", "📊 Overview", "🔄 Bulk Operations", "✅ Validation"])

# TAB 1: Edit Translations
with tabs[0]:
    st.markdown(f"### Editing: {languages[selected_lang]}")
    
    # Get all keys from English (master)
    all_keys = sorted(TRANSLATIONS['en'].keys())
    
    # Apply filters
    if search_term:
        all_keys = [k for k in all_keys if search_term.lower() in k.lower() or 
                    search_term.lower() in TRANSLATIONS['en'][k].lower()]
    
    if show_only_missing:
        all_keys = [k for k in all_keys if k not in TRANSLATIONS[selected_lang]]
    
    if show_only_untranslated:
        all_keys = [k for k in all_keys if k in TRANSLATIONS[selected_lang] and 
                    TRANSLATIONS[selected_lang][k] == TRANSLATIONS['en'][k]]
    
    st.info(f"Showing {len(all_keys)} keys")
    
    # Create editable table
    if all_keys:
        edit_data = []
        
        for key in all_keys:
            english_text = TRANSLATIONS['en'][key]
            current_text = TRANSLATIONS[selected_lang].get(key, "")
            
            # Status indicators
            if key not in TRANSLATIONS[selected_lang]:
                status = "❌ Missing"
            elif current_text == english_text:
                status = "⚠️ Untranslated"
            else:
                status = "✅ Translated"
            
            row = {
                'Key': key,
                'Status': status,
                'English': english_text,
                selected_lang.upper(): current_text
            }
            
            # Add comparison column if selected
            if compare_with != 'None':
                row[compare_with.upper()] = TRANSLATIONS[compare_with].get(key, "")
            
            edit_data.append(row)
        
        df = pd.DataFrame(edit_data)
        
        # Display with pagination
        items_per_page = st.selectbox("Items per page", [10, 25, 50, 100], index=1)
        total_pages = (len(df) - 1) // items_per_page + 1
        
        if 'page' not in st.session_state:
            st.session_state.page = 0
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Previous") and st.session_state.page > 0:
                st.session_state.page -= 1
        with col2:
            st.markdown(f"<div style='text-align: center'>Page {st.session_state.page + 1} of {total_pages}</div>", 
                       unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️") and st.session_state.page < total_pages - 1:
                st.session_state.page += 1
        
        start_idx = st.session_state.page * items_per_page
        end_idx = min(start_idx + items_per_page, len(df))
        
        page_df = df.iloc[start_idx:end_idx]
        
        st.dataframe(
            page_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Edit individual translation
        st.markdown("---")
        st.markdown("### ✏️ Edit Translation")
        
        selected_key = st.selectbox(
            "Select key to edit",
            options=all_keys,
            format_func=lambda k: f"{k} - {TRANSLATIONS['en'][k][:50]}..."
        )
        
        if selected_key:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**English (Reference)**")
                st.text_area(
                    "English",
                    value=TRANSLATIONS['en'][selected_key],
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
            
            with col_b:
                st.markdown(f"**{languages[selected_lang]}**")
                new_translation = st.text_area(
                    selected_lang,
                    value=TRANSLATIONS[selected_lang].get(selected_key, ""),
                    height=150,
                    label_visibility="collapsed"
                )
                
                if st.button("💾 Save Translation", type="primary"):
                    # Note: This is for demonstration only
                    # In production, you'd save to file or database
                    st.success(f"✅ Would save translation for '{selected_key}'")
                    st.info("ℹ️ In production, this would update translations.py file")
                    st.code(f"TRANSLATIONS['{selected_lang}']['{selected_key}'] = '''{new_translation}'''")

# TAB 2: Overview
with tabs[1]:
    st.markdown("### Translation Coverage Overview")
    
    # Create coverage matrix
    coverage_data = []
    
    for lang_code, lang_name in languages.items():
        lang_keys = set(TRANSLATIONS[lang_code].keys())
        missing = len(en_keys - lang_keys)
        translated = len(lang_keys)
        untranslated = sum(1 for k in lang_keys if TRANSLATIONS[lang_code][k] == TRANSLATIONS['en'].get(k, ''))
        completion = (translated / total_keys * 100) if total_keys > 0 else 0
        
        coverage_data.append({
            'Language': lang_name,
            'Code': lang_code,
            'Translated': translated,
            'Missing': missing,
            'Untranslated': untranslated,
            'Total': total_keys,
            'Completion %': f"{completion:.1f}%"
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    
    st.dataframe(
        coverage_df.style.background_gradient(
            subset=['Translated'],
            cmap='Greens'
        ).background_gradient(
            subset=['Missing'],
            cmap='Reds'
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # Visualization
    st.markdown("### 📊 Coverage Visualization")
    
    import plotly.express as px
    
    fig = px.bar(
        coverage_df,
        x='Language',
        y=['Translated', 'Missing', 'Untranslated'],
        title='Translation Status by Language',
        labels={'value': 'Number of Keys', 'variable': 'Status'},
        color_discrete_map={
            'Translated': '#28a745',
            'Missing': '#dc3545',
            'Untranslated': '#ffc107'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing keys by language
    st.markdown("### 🔍 Missing Keys by Language")
    
    selected_lang_overview = st.selectbox(
        "Select language to view missing keys",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        key='overview_lang'
    )
    
    missing_keys_list = list(en_keys - set(TRANSLATIONS[selected_lang_overview].keys()))
    
    if missing_keys_list:
        st.warning(f"Found {len(missing_keys_list)} missing keys")
        
        missing_df = pd.DataFrame([
            {'Key': k, 'English Text': TRANSLATIONS['en'][k]}
            for k in sorted(missing_keys_list)
        ])
        
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.success(f"✅ All keys translated for {languages[selected_lang_overview]}!")

# TAB 3: Bulk Operations
with tabs[2]:
    st.markdown("### 🔄 Bulk Operations")
    
    st.info("⚠️ These operations are for demonstration. In production, they would modify translations.py")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Copy Missing Keys")
        st.markdown("Copy missing keys from English (creates untranslated entries)")
        
        target_lang_copy = st.selectbox(
            "Target Language",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            key='copy_lang'
        )
        
        missing_count = len(en_keys - set(TRANSLATIONS[target_lang_copy].keys()))
        
        if st.button(f"Copy {missing_count} Missing Keys"):
            st.code(f"""
# This would add to translations.py:
for key in missing_keys:
    TRANSLATIONS['{target_lang_copy}'][key] = TRANSLATIONS['en'][key]
""")
            st.success(f"Would copy {missing_count} keys to {languages[target_lang_copy]}")
    
    with col2:
        st.markdown("#### 🗑️ Remove Unused Keys")
        st.markdown("Remove keys that don't exist in English (cleanup)")
        
        cleanup_lang = st.selectbox(
            "Language to Clean",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            key='cleanup_lang'
        )
        
        extra_keys = set(TRANSLATIONS[cleanup_lang].keys()) - en_keys
        
        if extra_keys:
            st.warning(f"Found {len(extra_keys)} extra keys")
            st.write(list(extra_keys)[:10])
            
            if st.button(f"Remove {len(extra_keys)} Extra Keys"):
                st.code(f"""
# This would remove from translations.py:
extra_keys = {extra_keys}
for key in extra_keys:
    del TRANSLATIONS['{cleanup_lang}'][key]
""")
                st.success(f"Would remove {len(extra_keys)} keys")
        else:
            st.success("✅ No extra keys found!")
    
    st.markdown("---")
    
    st.markdown("#### 🤖 Auto-Translate Missing Keys")
    st.markdown("Use Groq API to automatically translate missing keys")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        auto_translate_lang = st.selectbox(
            "Language to Auto-Translate",
            options=[k for k in languages.keys() if k != 'en'],
            format_func=lambda x: languages[x],
            key='auto_lang'
        )
    
    with col_b:
        groq_key = st.text_input("Groq API Key", type="password")
    
    missing_for_auto = len(en_keys - set(TRANSLATIONS[auto_translate_lang].keys()))
    
    if missing_for_auto > 0:
        if st.button(f"🚀 Auto-Translate {missing_for_auto} Keys"):
            if not groq_key:
                st.error("Please provide a Groq API key")
            else:
                st.info("This would use the translate_helper.py script to auto-translate missing keys")
                st.code(f"""
# Run this command:
python translate_helper.py --lang {auto_translate_lang} --api-key {groq_key[:10]}...
""")
    else:
        st.success(f"✅ All keys already translated for {languages[auto_translate_lang]}!")

# TAB 4: Validation
with tabs[3]:
    st.markdown("### ✅ Validation")
    
    validation_lang = st.selectbox(
        "Select Language to Validate",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        key='validate_lang'
    )
    
    st.markdown("---")
    
    # Run validations
    issues = []
    
    # Check 1: Missing keys
    missing = en_keys - set(TRANSLATIONS[validation_lang].keys())
    if missing:
        issues.append({
            'Type': 'Missing Keys',
            'Severity': 'Error',
            'Count': len(missing),
            'Description': f'{len(missing)} keys missing from translation'
        })
    
    # Check 2: Untranslated keys (same as English)
    untranslated = [k for k in TRANSLATIONS[validation_lang].keys() 
                    if TRANSLATIONS[validation_lang][k] == TRANSLATIONS['en'].get(k, '')]
    if untranslated:
        issues.append({
            'Type': 'Untranslated',
            'Severity': 'Warning',
            'Count': len(untranslated),
            'Description': f'{len(untranslated)} keys have same text as English'
        })
    
    # Check 3: Empty translations
    empty = [k for k in TRANSLATIONS[validation_lang].keys() 
             if not TRANSLATIONS[validation_lang][k].strip()]
    if empty:
        issues.append({
            'Type': 'Empty',
            'Severity': 'Error',
            'Count': len(empty),
            'Description': f'{len(empty)} keys have empty translations'
        })
    
    # Check 4: Format string mismatches
    format_mismatches = []
    for key in TRANSLATIONS[validation_lang].keys():
        if key in TRANSLATIONS['en']:
            en_text = TRANSLATIONS['en'][key]
            lang_text = TRANSLATIONS[validation_lang][key]
            
            # Count {variable} placeholders
            import re
            en_vars = set(re.findall(r'\{(\w+)\}', en_text))
            lang_vars = set(re.findall(r'\{(\w+)\}', lang_text))
            
            if en_vars != lang_vars:
                format_mismatches.append(key)
    
    if format_mismatches:
        issues.append({
            'Type': 'Format Mismatch',
            'Severity': 'Error',
            'Count': len(format_mismatches),
            'Description': f'{len(format_mismatches)} keys have mismatched format variables'
        })
    
    # Check 5: Length discrepancies (translation much longer/shorter than original)
    length_issues = []
    for key in TRANSLATIONS[validation_lang].keys():
        if key in TRANSLATIONS['en']:
            en_len = len(TRANSLATIONS['en'][key])
            lang_len = len(TRANSLATIONS[validation_lang][key])
            
            if en_len > 20:  # Only check for longer strings
                ratio = lang_len / en_len if en_len > 0 else 0
                if ratio > 2.0 or ratio < 0.5:
                    length_issues.append(key)
    
    if length_issues:
        issues.append({
            'Type': 'Length Discrepancy',
            'Severity': 'Warning',
            'Count': len(length_issues),
            'Description': f'{len(length_issues)} keys have unusual length differences (>2x or <0.5x)'
        })
    
    # Display results
    if issues:
        st.warning(f"Found {len(issues)} validation issues")
        
        issues_df = pd.DataFrame(issues)
        st.dataframe(
            issues_df.style.applymap(
                lambda x: 'background-color: #f8d7da' if x == 'Error' else 'background-color: #fff3cd',
                subset=['Severity']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Show details
        st.markdown("### Issue Details")
        
        issue_type = st.selectbox(
            "Select issue type to view details",
            options=[i['Type'] for i in issues]
        )
        
        if issue_type == 'Missing Keys' and missing:
            st.write(f"Missing keys ({len(missing)}):")
            st.write(sorted(list(missing))[:20])
            if len(missing) > 20:
                st.caption(f"... and {len(missing) - 20} more")
        
        elif issue_type == 'Untranslated' and untranslated:
            st.write(f"Untranslated keys ({len(untranslated)}):")
            untrans_df = pd.DataFrame([
                {'Key': k, 'Text': TRANSLATIONS['en'][k][:100]}
                for k in untranslated[:20]
            ])
            st.dataframe(untrans_df, use_container_width=True, hide_index=True)
        
        elif issue_type == 'Format Mismatch' and format_mismatches:
            st.write(f"Format mismatches ({len(format_mismatches)}):")
            mismatch_df = pd.DataFrame([
                {
                    'Key': k,
                    'English': TRANSLATIONS['en'][k],
                    validation_lang.upper(): TRANSLATIONS[validation_lang][k]
                }
                for k in format_mismatches[:10]
            ])
            st.dataframe(mismatch_df, use_container_width=True, hide_index=True)
        
        elif issue_type == 'Length Discrepancy' and length_issues:
            st.write(f"Length discrepancies ({len(length_issues)}):")
            length_df = pd.DataFrame([
                {
                    'Key': k,
                    'EN Length': len(TRANSLATIONS['en'][k]),
                    f'{validation_lang.upper()} Length': len(TRANSLATIONS[validation_lang][k]),
                    'Ratio': f"{len(TRANSLATIONS[validation_lang][k]) / len(TRANSLATIONS['en'][k]):.2f}x"
                }
                for k in length_issues[:20]
            ])
            st.dataframe(length_df, use_container_width=True, hide_index=True)
    
    else:
        st.success(f"✅ No validation issues found for {languages[validation_lang]}!")
        st.balloons()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Translation Manager • Part of Intrinsic Value Calculator Suite"
    "</div>",
    unsafe_allow_html=True
)