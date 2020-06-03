
const lineGraphMain = () => {

    const data_location = "./visualization_data/best_results.json";

    const svg = d3.select('#line-graph-svg');
    const lineGraphGroup = svg.append('g');
    const lineGraphTitle = lineGraphGroup.append('text');
    const attentionModelLines = lineGraphGroup.append('g');
    const plainRNNModelLines = lineGraphGroup.append('g');
    const xAxisGroup = lineGraphGroup.append('g');
    const xAxisLabel = xAxisGroup.append('text');
    const yAxisGroup = lineGraphGroup.append('g');
    const yAxisLabel = yAxisGroup.append('text');

    const margin = {
        top: 80,
        bottom: 80,
        left: 120,
        right: 30,
    };

    const attentionFill = "red";
    const plainRNNFill = "blue";
    
    const render = (attention_data, plain_rnn_data) => {
        
        const plotContainer = document.getElementById("line-graph");
        svg
            .attr('width', plotContainer.clientWidth)
            .attr('height', plotContainer.clientHeight);
        
        const svg_width = parseFloat(svg.attr('width'));
        const svg_height = parseFloat(svg.attr('height'));
        
        const innerWidth = svg_width - margin.left - margin.right;
        const innerHeight = svg_height - margin.top - margin.bottom;
        
        const labelSize = Math.min(20, innerWidth/40);
        lineGraphTitle
            .style('font-size', labelSize)
            .attr('x', innerWidth / 2)
            .attr('y', margin.top - labelSize * 2)
            .text("Validation Performance");
        
        const nameForAttentionModel = datum => `Attention Model #${datum.model_index}`;
        const nameForPlainRNNModel = datum => `RNN Model #${datum.model_index}`;
        const attentionNames = attention_data.map(nameForAttentionModel);
        const plainRNNNames = plain_rnn_data.map(nameForPlainRNNModel);
        const modelNames = attentionNames.concat(plainRNNNames);
        const modelMaxEpoch = data => Math.max(...data.validation_progress.map(datum => datum.epoch));
        const xMaxValue = Math.max(d3.max(attention_data, modelMaxEpoch), d3.max(plain_rnn_data, modelMaxEpoch));
        const xScale = d3.scaleLinear()
              .domain([0, xMaxValue])
              .range([0, innerWidth]);
        
        const yScale = d3.scaleLinear()
              .domain([1.0, 0.4])
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
        const mean = arr => arr.reduce((a,b) => a + b, 0) / arr.length;
        const tickTextAverageLength = mean(modelNames.map(name => name.length));
        xAxisGroup.call(d3.axisBottom(xScale).tickSize(-innerHeight).ticks(5))
            .attr('transform', `translate(${margin.left}, ${margin.top+innerHeight})`);
        xAxisGroup.selectAll('.tick line')
            .style('opacity', 0.1);
        xAxisGroup.selectAll('.tick text')
            .attr("transform", `translate(0, 10)`);
        xAxisLabel
            .style('font-size', labelSize * 0.8)
            .attr('fill', 'black')
            .attr('y', margin.bottom * 0.75)
            .attr('x', innerWidth / 2)
            .text('Number of Completed Epochs');

        attentionModelLines
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
        plainRNNModelLines
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
        
        const lineGenerator = d3.line()
              .x(datum => xScale(datum.epoch))
              .y(datum => yScale(datum.mean_loss));
        
        attentionModelLines.selectAll('path')
            .remove();
        attention_data.forEach(model => {
            attentionModelLines
                .append('path')
                .attr("class", "path")
                .attr("d", lineGenerator(model.validation_progress))
                .attr("fill", "none")
                .attr("stroke", attentionFill)
                .attr("stroke-width", 1.0);
        });

        plainRNNModelLines.selectAll('path')
            .remove();
        plain_rnn_data.forEach(model => {
            plainRNNModelLines
                .append('path')
                .attr("class", "path")
                .attr("d", lineGenerator(model.validation_progress))
                .attr("fill", "none")
                .attr("stroke", plainRNNFill)
                .attr("stroke-width", 1.0);
        });
                 
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
                        training_progress: [{epoch: 0, mean_loss: 1.0}].concat(datum.training_progress),
                        validation_progress: [{epoch: 0, mean_loss: 1.0}].concat(datum.validation_progress),
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

lineGraphMain();
