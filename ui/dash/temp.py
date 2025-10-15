        # # table from Yahoo Finance
        # # define URL
        # urlyahoo = 'https://finance.yahoo.com/losers'
        # tables = pd.read_html(urlyahoo)
        # losers = pd.DataFrame(tables[0])
        # print(losers.head())
        # # Drop the columns that do not matters
        # losers.drop(['Volume', 'Avg Vol (3 month)', 'PE Ratio (TTM)', '52 Week Range'], axis='columns', inplace=True)
        # losers.columns = ['Symbol','Company Name','Price', 'Change', '% Change', 'Mkt Cap']
        # # 2) add first column with empty boxes:
        # losers.insert(0, 'Select', '⬜') 

        # # create the tables to show the inforamtion
        # table = html.Div([
        #     dash_table.DataTable(
        #         columns=[{"name": i, "id": i} for i in losers.columns],
        #         data=losers.to_dict('records'),
        #         editable=False,
        #         style_as_list_view= True,
        #         style_data_conditional=[
        #             {'if': {'state': 'active'},'backgroundColor': 'white', 'border': '1px solid white'},
        #             {'if': {'column_id': 'Company Name'}, 'textAlign': 'left', 'text-indent': '10px', 'width':100},
        #             ],
        #         fixed_rows={'headers': True},
        #         id='table',
        #         style_data={"font-size" : "14px", 'width': 15, "background":"white", 'text-align': 'center'},
        #     )
        # ])


        # app = dash.Dash(__name__)

        # # Layout of the page:
        # app.layout = html.Div([
        #     html.H2("Today's Company Losers"),
        #     html.H4("Select a Symbol", id="Message1"),
        #     html.Div(table, style={'width':'60%'})
        # ])

        # # Callback
        # @app.callback(Output("Message1", "children"),
        #             Output("table", "data"),
        #             [Input('table', 'active_cell'),
        #             State('table', 'data')])
        # def update_loosers(cell,  data):
        #     # If there is not selection:
        #     if not cell:
        #         raise PreventUpdate
        #     else:
        #         # 3) If the user select a box of the "Select" column:
        #         if cell["column_id"] == 'Select':
        #             # takes info for some columns in the row selected
        #             symbol_selected = data[cell["row"]]["Symbol"]
        #             company_selected = data[cell["row"]]["Company Name"]
        #             message = "Last Symbol selected: - "+symbol_selected+" - Company Name:   "+company_selected
                    
        #             # 4) Change the figure of the box selected
        #             if data[cell["row"]]["Select"] == '⬜':
        #                 data[cell["row"]]["Select"] = '✅'
        #             else:
        #                 # 5) if the user unselect the selected box:
        #                 data[cell["row"]]["Select"] = '⬜'
        #                 message = "The Symbol: - "+symbol_selected+" - Company Name:   "+company_selected+" has been unselected"
                
        #         # if other column is selected do nothing:
        #         else:
        #             raise PreventUpdate

        #         return message, data