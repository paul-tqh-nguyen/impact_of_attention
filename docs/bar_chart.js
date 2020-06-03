
const mean = arr => arr.reduce((a,b) => a + b, 0) / arr.length;

const barChartMain = () => {

    const data_location = "./visualization_data/best_results.json";

    const svg = d3.select('#bar-chart-svg');
    const barChartGroup = svg.append('g');
    const barChartTitle = barChartGroup.append('text');
    const attentionModelBars = barChartGroup.append('g');
    const plainRNNModelBars = barChartGroup.append('g');
    const xAxisGroup = barChartGroup.append('g');
    const yAxisGroup = barChartGroup.append('g');
    const yAxisLabel = yAxisGroup.append('text');

    const margin = {
        top: 80,
        bottom: 20,
        left: 120,
        right: 30,
    };

    const attentionFill = "red";
    const plainRNNFill = "blue";
    
    const render = (attention_data, plain_rnn_data) => {
        
        const plotContainer = document.getElementById("bar-chart");
        svg
            .attr('width', plotContainer.clientWidth)
            .attr('height', plotContainer.clientHeight);
        
        const svg_width = parseFloat(svg.attr('width'));
        const svg_height = parseFloat(svg.attr('height'));
        
        const innerWidth = svg_width - margin.left - margin.right;
        const innerHeight = svg_height - margin.top - margin.bottom;

        const labelSize = Math.min(20, innerWidth/40);
        barChartTitle
            .style('font-size', labelSize)
            .attr('x', innerWidth * 0.5)
            .attr('y', margin.top - labelSize * 2)
            .text("Best Performing Models");

        let index2ModelName= {};
        const nameForAttentionModel = datum => `Attention Model #${datum.model_index}`;
        const nameForPlainRNNModel = datum => `RNN Model #${datum.model_index}`;
        const globalIndexForAttentionModel = datum => datum.model_index;
        const globalIndexForPlainRNNModel = datum => attention_data.length+datum.model_index;
        attention_data.forEach(datum => {index2ModelName[globalIndexForAttentionModel(datum)] = nameForAttentionModel(datum);});
        plain_rnn_data.forEach(datum => {index2ModelName[globalIndexForPlainRNNModel(datum)] = nameForPlainRNNModel(datum);});
        const modelNames = Object.values(index2ModelName);
        
        const xScale = d3.scaleBand()
              .domain(modelNames)
              .range([0, innerWidth]);
        
        const yScale = d3.scaleLinear()
              .domain([0.455, 0.4])
              .range([0, innerHeight]);
        
        yAxisGroup.call(d3.axisLeft(yScale).tickSize(-innerWidth))
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
        yAxisGroup.selectAll('.tick line')
            .attr('x', margin.left - 10)
            .style('opacity', 0.1);
        yAxisGroup.selectAll('.tick text')
            .attr('transform', 'translate(-10.0, 0.0)');
        yAxisLabel
            .style('font-size', labelSize * 0.8)
            .attr('fill', 'black')
            .attr("transform", "rotate(-90)")
            .attr('y', -60)
            .attr('x', -innerHeight/3)
            .text('Mean Cross Entropy Loss');
        
        const tickTextSize = Math.min(15, innerWidth/40);
        const tickTextAverageLength = mean(modelNames.map(name => name.length));
        xAxisGroup.call(d3.axisBottom(xScale).tickSize(-innerHeight))
            .attr('transform', `translate(${margin.left}, ${margin.top+innerHeight})`);
        xAxisGroup.selectAll('.tick line')
            .remove();
        xAxisGroup.selectAll('.tick text')
            .attr("transform", `translate(${-tickTextSize/2}, ${- tickTextAverageLength * tickTextSize * 0.4}) rotate(-90)`)
            .style('color', '#fff')
            .style('font-size', tickTextSize);
        
        attentionModelBars.selectAll('rect').data(attention_data)
            .remove();
        attentionModelBars.selectAll('rect').data(attention_data)
            .enter()
            .append('rect')
            .attr('y', datum => margin.top+yScale(datum['test_loss']))
            .attr('x', datum => xScale(nameForAttentionModel(datum))+margin.left)
            .attr('width', xScale.bandwidth())
            .attr('height', datum => innerHeight-yScale(datum['test_loss']))
            .attr('fill', attentionFill);
        
        plainRNNModelBars.selectAll('rect').data(plain_rnn_data)
            .remove();
        plainRNNModelBars.selectAll('rect').data(plain_rnn_data)
            .enter()
            .append('rect')
            .attr('y', datum => margin.top+yScale(datum['test_loss']))
            .attr('x', datum => xScale(nameForPlainRNNModel(datum))+margin.left)
            .attr('width', xScale.bandwidth())
            .attr('height', datum => innerHeight-yScale(datum['test_loss']))
            .attr('fill', plainRNNFill);
        
    };
    
    const redraw = () => {
        d3.json(data_location)
            .then(data => {
                let extract_data = (datum, datumIndex) => {
                    return {
                        model_index: datumIndex, 
                        number_of_epochs: parseInt(datum.number_of_epochs),
                        batch_size: parseInt(datum.batch_size),
                        vocab_size: parseInt(datum.vocab_size),
                        pre_trained_embedding_specification: datum.pre_trained_embedding_specification,
                        encoding_hidden_size: parseInt(datum.encoding_hidden_size),
                        number_of_encoding_layers: parseInt(datum.number_of_encoding_layers),
                        attention_intermediate_size: parseInt(datum.attention_intermediate_size),
                        number_of_attention_heads: parseInt(datum.number_of_attention_heads),
                        dropout_probability: parseFloat(datum.dropout_probability),
                        final_representation: datum.final_representation,
                        test_loss: parseFloat(datum.test_loss),
                        test_accuracy: parseFloat(datum.test_accuracy),
                        number_of_parameters: parseInt(datum.number_of_parameters),
                        // result_dir: datum.result_dir,
                    };
                };
                let attention_data = data['attention'].map(extract_data);
                let plain_rnn_data = data['plain_rnn'].map(extract_data);
                render(attention_data, plain_rnn_data);
            }).catch(err => {
                console.error(err.message);
                return;
            });
    };

    redraw();
    window.addEventListener("resize", redraw);

};

barChartMain();
